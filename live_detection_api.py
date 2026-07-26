import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from NVRConnect import (
    ACTIVE_CHANNELS,
    CAMERA_CONFIGS,
    NVR_IP,
    PersonTracker,
    open_camera,
    process_frame,
)
from detection_alert_db import get_all_alerts
from notification_logging import setup_notification_logging

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
PORT = 5051

MJPEG_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

_frame_lock = threading.Lock()
_raw_frames = {}
_display_frames = {}
_raw_updated_at = {}
_display_updated_at = {}
_stream_running = False


def _set_raw_frame(camera_id, frame):
    with _frame_lock:
        _raw_frames[camera_id] = cv2.resize(frame, (1280, 720))
        _raw_updated_at[camera_id] = time.time()


def _set_display_frame(camera_id, frame):
    with _frame_lock:
        _display_frames[camera_id] = cv2.resize(frame, (1280, 720))
        _display_updated_at[camera_id] = time.time()


def _pick_camera_frame(camera_id):
    display = _display_frames.get(camera_id)
    raw = _raw_frames.get(camera_id)
    if display is None:
        return None if raw is None else raw.copy()
    if raw is None:
        return display.copy()

    display_age = time.time() - _display_updated_at.get(camera_id, 0)
    if display_age > 1.0:
        return raw.copy()
    return display.copy()


def _get_stream_frame(camera_id=None):
    with _frame_lock:
        camera_ids = sorted(set(_raw_frames) | set(_display_frames))
        if camera_id is not None:
            frame = _pick_camera_frame(camera_id)
            return frame

        if not camera_ids:
            return None

        if len(camera_ids) == 1:
            return _pick_camera_frame(camera_ids[0])

        picked = [_pick_camera_frame(cid) for cid in camera_ids[:2]]
        if any(frame is None for frame in picked):
            return None

        f1 = cv2.resize(picked[0], (640, 360))
        f2 = cv2.resize(picked[1], (640, 360))
        return cv2.hconcat([f1, f2])


def _placeholder_frame(message="Connecting to cameras..."):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        message,
        (40, 360),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
    )
    return frame


def _encode_jpeg_frame(frame):
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return None
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
    )


def _read_latest_frame(cap):
    """Read from RTSP and drain the buffer so we always get the newest frame."""
    success, frame = cap.read()
    if not success or frame is None:
        return False, None

    for _ in range(3):
        grabbed, newer = cap.read()
        if grabbed and newer is not None:
            frame = newer
        else:
            break

    return True, frame


def _capture_loop(cap, name, camera_id):
    """Fast loop: only grab frames and push them to the live stream."""
    while _stream_running:
        success, frame = _read_latest_frame(cap)
        if success:
            _set_raw_frame(camera_id, frame)
        else:
            print(f"Failed to read from {name}")
            time.sleep(0.05)


def _process_loop(name, camera_id):
    """Slower loop: run PPE detection without blocking the live stream."""
    frame_counter = 0
    person_tracker = PersonTracker()

    while _stream_running:
        with _frame_lock:
            raw = None if camera_id not in _raw_frames else _raw_frames[camera_id].copy()

        if raw is None:
            time.sleep(0.05)
            continue

        frame_counter += 1
        processed = process_frame(raw, name, frame_counter, person_tracker)
        _set_display_frame(camera_id, processed)
        time.sleep(0.05)


def _start_single_camera(config, camera_id):
    def camera_worker():
        cap, name = open_camera(config)
        if cap is None:
            print(f"Skipping {config['name']} — could not connect.")
            return

        threading.Thread(
            target=_process_loop,
            args=(name, camera_id),
            daemon=True,
            name=f"process-{camera_id}",
        ).start()

        print(f"Live stream workers started for {name}.")
        _capture_loop(cap, name, camera_id)
        cap.release()

    threading.Thread(
        target=camera_worker,
        daemon=True,
        name=f"camera-{camera_id}",
    ).start()


def _start_camera_workers():
    global _stream_running

    _stream_running = True
    print(
        "Starting camera workers. Close NVRConnect.py first — "
        "the NVR allows only one RTSP client per channel."
    )

    for i, config in enumerate(CAMERA_CONFIGS, start=1):
        _start_single_camera(config, i)


def _generate_mjpeg(camera_id=None):
    placeholder = _encode_jpeg_frame(_placeholder_frame())
    if placeholder is not None:
        yield placeholder

    while True:
        frame = _get_stream_frame(camera_id)
        if frame is None:
            placeholder = _encode_jpeg_frame(_placeholder_frame())
            if placeholder is not None:
                yield placeholder
            time.sleep(0.5)
            continue

        chunk = _encode_jpeg_frame(frame)
        if chunk is None:
            continue

        yield chunk
        time.sleep(0.033)


@app.route("/")
def index():
    camera_blocks = "".join(
        f"<div><h3>{config['name']}</h3>"
        f"<p style='margin:0 0 8px;color:#aaa'>{config.get('location', 'Unknown')}</p>"
        f"<img src='/live-detection-camera-{i}' "
        f"style='width:100%;max-width:640px;display:block'/></div>"
        for i, config in enumerate(CAMERA_CONFIGS, start=1)
    )
    return (
        "<html><body style='margin:0;background:#111;color:#fff;font-family:sans-serif'>"
        f"<h2 style='padding:16px'>PPE Live Detection — NVR ({NVR_IP})</h2>"
        "<div style='display:flex;flex-wrap:wrap;gap:16px;justify-content:center;padding:16px'>"
        f"{camera_blocks}"
        "</div></body></html>"
    )


@app.route("/live-detection")
def live_detection():
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers=MJPEG_HEADERS,
    )


@app.route("/live-detection-camera-<int:camera_id>")
def live_detection_camera(camera_id):
    if camera_id < 1 or camera_id > len(CAMERA_CONFIGS):
        return jsonify({"error": f"Camera {camera_id} not configured"}), 404
    return Response(
        _generate_mjpeg(camera_id=camera_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers=MJPEG_HEADERS,
    )


@app.route("/health")
def health():
    return {
        "status": "ok",
        "stream_running": _stream_running,
        "nvr_ip": NVR_IP,
        "active_channels": ACTIVE_CHANNELS,
        "cameras": {
            f"camera_{i}": {
                "name": config["name"],
                "location": config.get("location", "Unknown"),
                "has_frame": i in _raw_frames or i in _display_frames,
                "endpoint": f"/live-detection-camera-{i}",
            }
            for i, config in enumerate(CAMERA_CONFIGS, start=1)
        },
    }


@app.route("/detection-alerts", methods=["GET"])
def detection_alerts():
    status = request.args.get("status")
    alerts = get_all_alerts(status=status)
    return jsonify({"count": len(alerts), "data": alerts})


def main():
    log_file = setup_notification_logging()
    print(f"Notification logs: {log_file}")

    threading.Thread(target=_start_camera_workers, daemon=True).start()

    print(f"API running on http://0.0.0.0:{PORT}")
    print(f"NVR: {NVR_IP} — {len(CAMERA_CONFIGS)} channel(s) configured")
    for i, config in enumerate(CAMERA_CONFIGS, start=1):
        print(
            f"  Camera {i} ({config.get('location', 'Unknown')}): "
            f"http://localhost:{PORT}/live-detection-camera-{i}"
        )
    print(f"Combined: http://localhost:{PORT}/live-detection")
    print(f"Alerts: http://localhost:{PORT}/detection-alerts")
    app.run(host="0.0.0.0", port=PORT, threaded=True)


if __name__ == "__main__":
    main()
