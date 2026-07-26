import os
import threading
import time

import cv2
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from NVRConnect import (
    CAMERA_CONFIGS,
    NVR_CONFIGS,
    PersonTracker,
    get_nvr_summary,
    open_camera,
    process_frame,
)
from detection_alert_db import get_all_alerts
from notification_logging import setup_notification_logging

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
PORT = 5051

_frame_lock = threading.Lock()
_latest_frames = {}
_stream_running = False

_CAMERA_BY_ID = {cam["id"]: cam for cam in CAMERA_CONFIGS}
_NVR_CAMERA_LOOKUP = {
    (cam["nvr_id"], cam["camera_index"]): cam["id"] for cam in CAMERA_CONFIGS
}


def _camera_endpoint(camera_id):
    return f"/live-detection-camera-{camera_id}"


def _nvr_camera_endpoint(nvr_id, camera_index):
    return f"/live-detection/{nvr_id}/camera-{camera_index}"


def _connect_cameras():
    caps = []
    for config in CAMERA_CONFIGS:
        cap, name = open_camera(config)
        if cap is not None:
            caps.append((cap, name, config["id"]))
    return caps


def _camera_loop():
    global _latest_frames, _stream_running

    caps = _connect_cameras()
    if not caps:
        print("No cameras connected. Stream will be unavailable.")
        _stream_running = False
        return

    _stream_running = True
    frame_counters = {camera_id: 0 for _, _, camera_id in caps}
    person_trackers = {camera_id: PersonTracker() for _, _, camera_id in caps}

    print(f"Live detection stream started with {len(caps)} camera(s) across {len(NVR_CONFIGS)} NVR(s).")

    while _stream_running:
        for cap, name, camera_id in caps:
            success, frame = cap.read()
            if success:
                frame_counters[camera_id] += 1
                processed = process_frame(
                    frame, name, frame_counters[camera_id], person_trackers[camera_id]
                )
                with _frame_lock:
                    _latest_frames[camera_id] = cv2.resize(processed, (1280, 720))
            else:
                print(f"Failed to read from {name}")

    for cap, _, _ in caps:
        cap.release()


def _generate_mjpeg(camera_id=None):
    while True:
        with _frame_lock:
            if camera_id is not None:
                frame = None if camera_id not in _latest_frames else _latest_frames[camera_id].copy()
            elif _latest_frames:
                frames = sorted(_latest_frames.items())
                if len(frames) == 1:
                    frame = frames[0][1].copy()
                else:
                    f1 = cv2.resize(frames[0][1], (640, 360))
                    f2 = cv2.resize(frames[1][1], (640, 360))
                    frame = cv2.hconcat([f1, f2])
            else:
                frame = None

        if frame is None:
            time.sleep(0.1)
            continue

        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
        time.sleep(0.033)


def _camera_public(cam, has_frame=False):
    return {
        "id": cam["id"],
        "name": cam["name"],
        "location": cam.get("location", "Unknown"),
        "nvr_id": cam["nvr_id"],
        "nvr_name": cam["nvr_name"],
        "nvr_ip": cam["nvr_ip"],
        "camera_index": cam["camera_index"],
        "channel": cam.get("channel"),
        "has_frame": has_frame or cam["id"] in _latest_frames,
        "endpoint": _camera_endpoint(cam["id"]),
        "nvr_endpoint": _nvr_camera_endpoint(cam["nvr_id"], cam["camera_index"]),
    }


@app.route("/")
def index():
    nvr_sections = []
    for nvr in get_nvr_summary():
        camera_blocks = "".join(
            f"<div><h4>{cam['name']}</h4>"
            f"<p style='margin:0 0 8px;color:#aaa'>{cam.get('location', 'Unknown')}</p>"
            f"<img src='{_camera_endpoint(cam['id'])}' "
            f"style='width:100%;max-width:480px;display:block'/></div>"
            for cam in nvr["cameras"]
        )
        nvr_sections.append(
            f"<section style='margin-bottom:32px'>"
            f"<h3>{nvr['name']} ({nvr['ip']}) — {nvr['camera_count']} camera(s)</h3>"
            f"<div style='display:flex;flex-wrap:wrap;gap:16px'>{camera_blocks}</div>"
            f"</section>"
        )

    return (
        "<html><body style='margin:0;background:#111;color:#fff;font-family:sans-serif'>"
        f"<h2 style='padding:16px'>PPE Live Detection — {len(NVR_CONFIGS)} NVR(s), "
        f"{len(CAMERA_CONFIGS)} camera(s)</h2>"
        "<div style='padding:16px'>"
        f"{''.join(nvr_sections)}"
        "</div></body></html>"
    )


@app.route("/live-detection")
def live_detection():
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/live-detection-camera-<int:camera_id>")
def live_detection_camera(camera_id):
    if camera_id not in _CAMERA_BY_ID:
        return jsonify({"error": f"Camera {camera_id} not configured"}), 404
    return Response(
        _generate_mjpeg(camera_id=camera_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/live-detection/<nvr_id>/camera-<int:camera_index>")
def live_detection_nvr_camera(nvr_id, camera_index):
    camera_id = _NVR_CAMERA_LOOKUP.get((nvr_id, camera_index))
    if camera_id is None:
        return jsonify({"error": f"Camera {camera_index} not found on NVR '{nvr_id}'"}), 404
    return Response(
        _generate_mjpeg(camera_id=camera_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/nvrs")
def api_nvrs():
    nvrs = get_nvr_summary()
    return jsonify(
        {
            "count": len(nvrs),
            "data": [
                {
                    "id": nvr["id"],
                    "name": nvr["name"],
                    "ip": nvr["ip"],
                    "brand": nvr["brand"],
                    "rtsp_port": nvr["rtsp_port"],
                    "camera_count": nvr["camera_count"],
                    "cameras": [
                        _camera_public(cam) for cam in nvr["cameras"]
                    ],
                }
                for nvr in nvrs
            ],
        }
    )


@app.route("/api/nvrs/<nvr_id>")
def api_nvr_detail(nvr_id):
    nvrs = {nvr["id"]: nvr for nvr in get_nvr_summary()}
    nvr = nvrs.get(nvr_id)
    if nvr is None:
        return jsonify({"error": f"NVR '{nvr_id}' not configured"}), 404
    return jsonify(
        {
            "id": nvr["id"],
            "name": nvr["name"],
            "ip": nvr["ip"],
            "brand": nvr["brand"],
            "rtsp_port": nvr["rtsp_port"],
            "camera_count": nvr["camera_count"],
            "cameras": [_camera_public(cam) for cam in nvr["cameras"]],
        }
    )


@app.route("/api/cameras")
def api_cameras():
    return jsonify(
        {
            "count": len(CAMERA_CONFIGS),
            "data": [_camera_public(cam) for cam in CAMERA_CONFIGS],
        }
    )


@app.route("/health")
def health():
    return {
        "status": "ok",
        "stream_running": _stream_running,
        "nvr_count": len(NVR_CONFIGS),
        "camera_count": len(CAMERA_CONFIGS),
        "connected_frames": len(_latest_frames),
        "nvrs": [
            {
                "id": nvr["id"],
                "name": nvr["name"],
                "ip": nvr["ip"],
                "brand": nvr.get("brand"),
                "camera_count": sum(1 for c in CAMERA_CONFIGS if c["nvr_id"] == nvr["id"]),
            }
            for nvr in NVR_CONFIGS
        ],
        "cameras": {
            f"camera_{cam['id']}": _camera_public(cam) for cam in CAMERA_CONFIGS
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

    thread = threading.Thread(target=_camera_loop, daemon=True)
    thread.start()

    print(f"API running on http://0.0.0.0:{PORT}")
    print(f"{len(NVR_CONFIGS)} NVR(s), {len(CAMERA_CONFIGS)} camera(s) configured")
    for nvr in get_nvr_summary():
        print(f"  NVR {nvr['id']}: {nvr['name']} ({nvr['ip']}) — {nvr['camera_count']} camera(s)")
        for cam in nvr["cameras"]:
            print(
                f"    Camera {cam['id']} [{cam['nvr_id']}/camera-{cam['camera_index']}]: "
                f"http://localhost:{PORT}{_camera_endpoint(cam['id'])}"
            )
    print(f"Combined: http://localhost:{PORT}/live-detection")
    print(f"NVR list: http://localhost:{PORT}/api/nvrs")
    print(f"Alerts: http://localhost:{PORT}/detection-alerts")
    app.run(host="0.0.0.0", port=PORT, threaded=True)


if __name__ == "__main__":
    main()
