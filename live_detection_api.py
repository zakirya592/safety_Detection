import os
import threading
import time

import cv2
from flask import Flask, Response
from flask_cors import CORS

# Import detection logic and models from camera_shoes
from camera_shoes import CAMERA_CONFIGS, PersonTracker, process_frame

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
PORT = 5051

_frame_lock = threading.Lock()
_latest_combined_frame = None
_stream_running = False


def _connect_cameras():
    caps = []
    for config in CAMERA_CONFIGS:
        print(f"Connecting to {config['name']} at {config['ip']}...")
        cap = None
        for rtsp_url in config["rtsp_urls"]:
            print(f"  Trying: {rtsp_url}")
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("H", "2", "6", "4"))

            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    caps.append((cap, config["name"]))
                    print(f"Connected to {config['name']}")
                    break
                cap.release()
            else:
                cap.release()

        if cap is None or not any(c[0] == cap for c in caps):
            print(f"Failed to connect to {config['name']}")

    return caps


def _camera_loop():
    global _latest_combined_frame, _stream_running

    caps = _connect_cameras()
    if not caps:
        print("No cameras connected. Stream will be unavailable.")
        _stream_running = False
        return

    _stream_running = True
    frame_counters = {name: 0 for _, name in caps}
    person_trackers = {name: PersonTracker() for _, name in caps}

    print(f"Live detection stream started with {len(caps)} camera(s).")

    while _stream_running:
        frames = []
        for cap, name in caps:
            success, frame = cap.read()
            if success:
                frame_counters[name] += 1
                processed = process_frame(
                    frame, name, frame_counters[name], person_trackers[name]
                )
                frames.append(processed)
            else:
                print(f"Failed to read from {name}")

        if frames:
            if len(frames) == 1:
                combined = cv2.resize(frames[0], (1280, 720))
            else:
                f1 = cv2.resize(frames[0], (640, 360))
                f2 = cv2.resize(frames[1], (640, 360))
                combined = cv2.hconcat([f1, f2])

            with _frame_lock:
                _latest_combined_frame = combined

    for cap, _ in caps:
        cap.release()


def _generate_mjpeg():
    while True:
        with _frame_lock:
            frame = None if _latest_combined_frame is None else _latest_combined_frame.copy()

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


@app.route("/")
def index():
    return (
        "<html><body style='margin:0;background:#111;color:#fff;font-family:sans-serif'>"
        "<h2 style='padding:16px'>PPE Live Detection</h2>"
        f"<img src='/live-detection' style='width:100%;max-width:1280px;display:block;margin:auto'/>"
        "</body></html>"
    )


@app.route("/live-detection")
def live_detection():
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/health")
def health():
    return {
        "status": "ok",
        "stream_running": _stream_running,
        "has_frame": _latest_combined_frame is not None,
    }


def main():
    thread = threading.Thread(target=_camera_loop, daemon=True)
    thread.start()

    print(f"API running on http://0.0.0.0:{PORT}")
    print(f"Live stream: http://localhost:{PORT}/live-detection")
    app.run(host="0.0.0.0", port=PORT, threaded=True)


if __name__ == "__main__":
    main()
