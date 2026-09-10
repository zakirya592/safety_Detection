import os
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

from alarm import Alarm
from screenshot import ScreenshotManager
from detection_alert_db import save_detection_alerts_async
from unifi_discover import fetch_snapshot_jpeg

# Initialize alarm
alarm = Alarm()
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

RTSP_OPEN_TIMEOUT_MS = 2500
_working_rtsp_urls = {}
_open_camera_sema = threading.Semaphore(4)
_inference_lock = threading.Lock()

# ===========================================
# Parallel Camera Processing
# ===========================================

latest_frames = {}
frame_lock = threading.Lock()

running = True

# Initialize screenshot manager with 30-second reset time
screenshot_manager = ScreenshotManager(reset_time_seconds=15)

# Load both YOLO models
boots_model = YOLO("bestsss.pt")
ppe_model = YOLO("best.pt")

# Class mapping for the boots model
BOOTS_CLASSES = {
    0: "glove",
    1: "goggles",
    2: "helmet",
    3: "mask",
    4: "no_glove",
    5: "no_goggles",
    6: "no_helmet",
    7: "no_mask",
}

# Class mapping for the PPE model
PPE_CLASSES = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "Machinery",
    9: "Vehicle"
}

# ---------------------------------------------------------------------------
# PPE item classes we care about for the person-level compliance decision.
# "positive" = the item IS being worn. "negative" = the model explicitly says
# it is NOT being worn. Anything not seen at all for a person is left as
# "unknown" rather than assumed compliant or a violation.
# ---------------------------------------------------------------------------
HELMET_POSITIVE = {"Hardhat", "helmet"}
HELMET_NEGATIVE = {"NO-Hardhat", "no_helmet"}
VEST_POSITIVE = {"Safety Vest", "vest"}
VEST_NEGATIVE = {"NO-Safety Vest"}
GLOVE_POSITIVE = {"glove", "gloves"}
GLOVE_NEGATIVE = {"no_glove", "no_gloves"}

GOGGLES_POSITIVE = {"goggles"}
GOGGLES_NEGATIVE = {"no_goggles", "no_goggle"}

# All the item labels we bother drawing/considering (Person is handled separately)
ITEM_LABELS = HELMET_POSITIVE | HELMET_NEGATIVE | VEST_POSITIVE | VEST_NEGATIVE | GLOVE_POSITIVE | GLOVE_NEGATIVE | GOGGLES_POSITIVE | GOGGLES_NEGATIVE


# Confidence threshold for Person class only (lowered to 30% to detect more people)
PERSON_CONFIDENCE_THRESHOLD = 0.30

# Fraction of an item's own box area that must fall inside a person's box
# for that item to be considered "worn by" that person.
ITEM_CONTAINMENT_THRESHOLD = 0.5

# Performance optimization settings
PROCESS_EVERY_N_FRAMES = 8
PERSON_INPUT_SIZE = 640
PPE_INPUT_SIZE = 640
BOOTS_INPUT_SIZE = 640
PPE_ITEM_CONFIDENCE = 0.25
BOOTS_ITEM_CONFIDENCE = 0.10

# Person tracking settings
MAX_MISSING_FRAMES = 10  # Remove tracked person after 10 consecutive frames without detection
IOU_THRESHOLD = 0.3       # Intersection over Union threshold for matching detections to tracks

RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)


def _model_names(model):
    names = getattr(model, "names", None) or {}
    if isinstance(names, dict):
        return {int(k): str(v).strip() for k, v in names.items()}
    return {i: str(v).strip() for i, v in enumerate(names)}


def _class_ids_named(model, label):
    wanted = str(label).strip().lower()
    return [idx for idx, name in _model_names(model).items() if name.lower() == wanted]


def _is_person_label(label):
    return str(label).strip().lower() == "person"


def _label_for(model, class_id, class_map):
    names = _model_names(model)
    if class_id in names and names[class_id]:
        return names[class_id]
    return class_map.get(class_id, str(class_id))


def _collect_detections(results, model, class_map, min_conf=0.0, person_min_conf=PERSON_CONFIDENCE_THRESHOLD):
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = _label_for(model, class_id, class_map)
            if _is_person_label(label):
                if confidence < person_min_conf:
                    continue
            elif confidence < min_conf:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "box": [x1, y1, x2, y2],
                "label": label,
                "confidence": confidence,
            })
    return detections


def _expand_box(box, frame_shape, pad_ratio=0.2):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    ]


def _shift_box(box, ox, oy):
    return [box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy]


def _dedupe_persons(person_detections):
    deduped = []
    used = set()
    for i, p1 in enumerate(person_detections):
        if i in used:
            continue
        group = [i]
        for j, p2 in enumerate(person_detections):
            if j <= i or j in used:
                continue
            if calculate_iou(p1["box"], p2["box"]) > 0.5:
                group.append(j)
        best = max(group, key=lambda idx: person_detections[idx]["confidence"])
        deduped.append(person_detections[best])
        used.update(group)
    return deduped


def detect_persons(frame):
    """Step 1: find people first. PPE is not run until this returns someone."""
    predict_kwargs = {
        "source": frame,
        "imgsz": PERSON_INPUT_SIZE,
        "conf": PERSON_CONFIDENCE_THRESHOLD,
        "verbose": False,
    }
    person_ids = _class_ids_named(ppe_model, "Person")
    if person_ids:
        predict_kwargs["classes"] = person_ids
    with _inference_lock:
        results = ppe_model.predict(**predict_kwargs)
    persons = [d for d in _collect_detections(results, ppe_model, PPE_CLASSES) if _is_person_label(d["label"])]
    return _dedupe_persons(persons)


def detect_ppe_for_persons(frame, person_boxes):
    """Step 2: only after a person is found, inspect PPE on that person."""
    items = []
    for person_box in person_boxes:
        x1, y1, x2, y2 = _expand_box(person_box, frame.shape, 0.2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
            continue

        with _inference_lock:
            ppe_results = ppe_model.predict(
                crop, imgsz=PPE_INPUT_SIZE, conf=PPE_ITEM_CONFIDENCE, verbose=False
            )
            boots_results = boots_model.predict(
                crop, imgsz=BOOTS_INPUT_SIZE, conf=BOOTS_ITEM_CONFIDENCE, verbose=False
            )
        for det in _collect_detections(ppe_results, ppe_model, PPE_CLASSES, min_conf=PPE_ITEM_CONFIDENCE):
            if _is_person_label(det["label"]) or det["label"] not in ITEM_LABELS:
                continue
            det["box"] = _shift_box(det["box"], x1, y1)
            items.append(det)

        for det in _collect_detections(boots_results, boots_model, BOOTS_CLASSES, min_conf=BOOTS_ITEM_CONFIDENCE):
            if det["label"] not in ITEM_LABELS:
                continue
            det["box"] = _shift_box(det["box"], x1, y1)
            items.append(det)
    return items


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def containment_ratio(inner_box, outer_box):
    """
    Fraction of inner_box's own area that lies inside outer_box.
    Used to decide whether a small item box (Hardhat, vest, ...) belongs
    to a given person's (much bigger) box.
    """
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box

    x1 = max(ix1, ox1)
    y1 = max(iy1, oy1)
    x2 = min(ix2, ox2)
    y2 = min(iy2, oy2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    inner_area = (ix2 - ix1) * (iy2 - iy1)

    if inner_area <= 0:
        return 0.0

    return inter / inner_area


def classify_person_ppe(person_box, item_detections):
    """
    Given one person's box and every item detection from this frame,
    decide the person's Hardhat/vest status.

    Returns a dict:
        {
            'Hardhat': 'present' | 'missing' | 'unknown',
            'vest':   'present' | 'missing' | 'unknown',
            'gloves': 'present' | 'missing' | 'unknown',
            'goggles': 'present' | 'missing' | 'unknown',
            'missing_items': [...],
            'is_violation': bool,
            'is_fully_compliant': bool,
            'label': str,
        }
    """
    helmet_positive_seen = False
    helmet_negative_seen = False
    vest_positive_seen = False
    vest_negative_seen = False

    glove_positive_seen = False
    glove_negative_seen = False
    goggles_positive_seen = False
    goggles_negative_seen = False

    for item in item_detections:
        if containment_ratio(item['box'], person_box) < ITEM_CONTAINMENT_THRESHOLD:
            continue

        label = item['label']
        if label in HELMET_POSITIVE:
            helmet_positive_seen = True
        elif label in HELMET_NEGATIVE:
            helmet_negative_seen = True
        elif label in VEST_POSITIVE:
            vest_positive_seen = True
        elif label in VEST_NEGATIVE:
            vest_negative_seen = True

        elif label in GLOVE_POSITIVE:
            glove_positive_seen = True
        elif label in GLOVE_NEGATIVE:
            glove_negative_seen = True
        elif label in GOGGLES_POSITIVE:
            goggles_positive_seen = True
        elif label in GOGGLES_NEGATIVE:
            goggles_negative_seen = True
        

    # An explicit "NO-..." detection always wins over a positive one for
    # the same item, since the model is actively flagging a violation.
    if helmet_negative_seen:
        helmet_status = "missing"
    elif helmet_positive_seen:
        helmet_status = "present"
    else:
        helmet_status = "unknown"

    if vest_negative_seen:
        vest_status = "missing"
    elif vest_positive_seen:
        vest_status = "present"
    else:
        vest_status = "unknown"
    
    if glove_negative_seen:
        glove_status = "missing"
    elif glove_positive_seen:
        glove_status = "present"
    else:
        glove_status = "unknown"

    if goggles_negative_seen:
        goggles_status = "missing"
    elif goggles_positive_seen:
        goggles_status = "present"
    else:
        goggles_status = "unknown"
    

    missing_items = []
    if helmet_status == "missing":
        missing_items.append("Hardhat")
    if vest_status == "missing":
        missing_items.append("Vest")

    if glove_status == "missing":
        missing_items.append("Gloves")
    if goggles_status == "missing":
        missing_items.append("Goggles")

    is_violation = len(missing_items) > 0
    is_fully_compliant = (helmet_status == "present" and vest_status == "present" and glove_status == "present" and goggles_status == "present")

    if is_violation:
        label_text = "Missing " + " , ".join(missing_items)
    elif is_fully_compliant:
        label_text = "Hardhat + Vest + Gloves + Goggles OK"
    else:
        label_text = "Person"

    return {
        "Hardhat": helmet_status,
        "vest": vest_status,
        "gloves": glove_status,
        "goggles": goggles_status,
        "missing_items": missing_items,
        "is_violation": is_violation,
        "is_fully_compliant": is_fully_compliant,
        "label": label_text,
    }


class PersonTracker:
    """Track persons across frames to maintain persistent boxes + PPE status"""

    def __init__(self):
        self.tracks = {}  # {track_id: {'box', 'missing_frames', 'confidence', 'status'}}
        self.next_id = 0

    def calculate_iou(self, box1, box2):
        return calculate_iou(box1, box2)

    def update(self, detected_persons):
        """
        Update tracks with new detections.
        detected_persons: list of {'box': [x1, y1, x2, y2], 'confidence': float, 'status': dict}
        Returns: list of all active tracks with their boxes, confidence and PPE status
        """
        for track_id in self.tracks:
            self.tracks[track_id]['missing_frames'] += 1

        matched_track_ids = set()

        for detection in detected_persons:
            detection_box = detection['box']
            best_iou = 0
            best_track_id = None

            for track_id, track in self.tracks.items():
                if track_id in matched_track_ids:
                    continue

                iou = self.calculate_iou(detection_box, track['box'])
                if iou > best_iou and iou > IOU_THRESHOLD:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                self.tracks[best_track_id]['box'] = detection_box
                self.tracks[best_track_id]['missing_frames'] = 0
                self.tracks[best_track_id]['confidence'] = detection['confidence']
                self.tracks[best_track_id]['status'] = detection['status']
                matched_track_ids.add(best_track_id)
            else:
                self.tracks[self.next_id] = {
                    'box': detection_box,
                    'missing_frames': 0,
                    'confidence': detection['confidence'],
                    'status': detection['status'],
                }
                matched_track_ids.add(self.next_id)
                self.next_id += 1

        tracks_to_remove = [
            track_id for track_id, track in self.tracks.items()
            if track['missing_frames'] > MAX_MISSING_FRAMES
        ]
        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        return [
            {
                'track_id': track_id,
                'box': track['box'],
                'confidence': track['confidence'],
                'status': track['status'],
            }
            for track_id, track in self.tracks.items()
        ]


# ---------------------------------------------------------------------------
# Camera / NVR configuration (three UniFi NVRs from .env / unifi_cameras.json)
# ---------------------------------------------------------------------------
from nvr_config import (
    ACTIVE_CHANNELS,
    CAMERA_CONFIGS,
    NVR_CONFIGS,
    NVR_IP,
    PASS_ENC,
    RAW_PASSWORD,
    RAW_USERNAME,
    build_hikvision_rtsp_urls as build_rtsp_urls,
    build_unifi_rtsp_urls,
    get_nvr_summary,
)

CAMERA_LOCATIONS = {config["name"]: config.get("location", "Unknown") for config in CAMERA_CONFIGS}


def draw_person_box(annotated, box, status, confidence, track_id=None):
    """
    Draws ONE box per person (green if compliant/unknown, red if a
    violation was found), a dimensions readout, and the PPE status label.
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    color = RED if status["is_violation"] else GREEN

    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 5)

    id_part = f"ID{track_id} " if track_id is not None else ""
    dims_text = f"{id_part}W:{width} H:{height} ({confidence:.2f})"
    cv2.putText(annotated, dims_text, (x1, max(15, y1 - 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(annotated, status["label"], (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return width, height


def process_frame(frame, camera_name, frame_count, person_tracker):
    """
    Person-first pipeline:
      1. Detect people with the PPE model (the only model that has Person).
      2. If nobody is found, skip PPE and keep waiting.
      3. If a person is found, run helmet/vest/gloves/goggles on that person only.
    """
    violating_persons = []
    annotated = frame

    if frame_count % PROCESS_EVERY_N_FRAMES != 0:
        for track_id, track in person_tracker.tracks.items():
            draw_person_box(annotated, track["box"], track["status"],
                             track["confidence"], track_id)
        cv2.putText(annotated, camera_name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return annotated

    person_detections = detect_persons(frame)

    if not person_detections:
        alarm.stop()
        person_tracker.update([])
        cv2.putText(annotated, camera_name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated, "No person — PPE waiting", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2)
        return annotated

    item_detections = detect_ppe_for_persons(
        frame, [person["box"] for person in person_detections]
    )

    detected_persons = []
    for person in person_detections:
        status = classify_person_ppe(person["box"], item_detections)
        detected_persons.append({
            "box": person["box"],
            "confidence": person["confidence"],
            "status": status,
        })

        if status["is_violation"]:
            violating_persons.append({
                "label": " & ".join(status["missing_items"]),
                "x1": person["box"][0], "y1": person["box"][1],
                "x2": person["box"][2], "y2": person["box"][3],
                "confidence": person["confidence"],
            })

    if violating_persons:
        alarm.play()
        print(f"Violations detected: {[p['label'] for p in violating_persons]}")
        print(f"Violating persons count: {len(violating_persons)}")
        screenshot_result = screenshot_manager.take_screenshot(
            frame, violating_persons, camera_name=camera_name
        )
        if screenshot_result:
            print(f"Screenshot saved: {screenshot_result['path']}")
            save_detection_alerts_async(
                camera=camera_name,
                location=CAMERA_LOCATIONS.get(camera_name, "Unknown"),
                persons=screenshot_result["persons"],
                image_url=screenshot_result.get("image_url"),
            )
        else:
            print("Screenshot not saved (possibly already photographed)")
    else:
        alarm.stop()

    active_tracks = person_tracker.update(detected_persons)

    for track in active_tracks:
        draw_person_box(annotated, track["box"], track["status"],
                         track["confidence"], track["track_id"])

    cv2.putText(annotated, camera_name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(
        annotated,
        f"Persons: {len(detected_persons)} | PPE on",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        GREEN,
        2,
    )

    return annotated


def camera_worker(config):
    """
    Each camera runs independently inside its own thread.
    """

    global latest_frames
    global running

    cap, camera_name = open_camera(config)

    if cap is None:
        print(f"Unable to connect {config['name']}")
        return

    frame_counter = 0

    tracker = PersonTracker()

    print(f"{camera_name} thread started")

    while running:

        cap.grab()
        success, frame = cap.read()

        if not success:

            print(f"{camera_name} disconnected")

            cap.release()

            while running:

                print(f"Trying reconnect {camera_name}")

                cap, _ = open_camera(config)

                if cap is not None:

                    print(f"{camera_name} reconnected")

                    break

                time.sleep(5)

            continue

        frame_counter += 1

        processed = process_frame(
            frame,
            camera_name,
            frame_counter,
            tracker
        )

        with frame_lock:
            latest_frames[camera_name] = processed

    cap.release()

    print(f"{camera_name} thread stopped")


class ProtectSnapshotCapture:
    """OpenCV-like capture that pulls JPEG snapshots from UniFi Protect."""

    is_snapshot = True

    def __init__(self, nvr_ip, protect_id, name):
        self.nvr_ip = nvr_ip
        self.protect_id = protect_id
        self.name = name
        self._opened = True

    def isOpened(self):
        return self._opened

    def read(self):
        jpeg = fetch_snapshot_jpeg(self.nvr_ip, self.protect_id)
        if not jpeg:
            return False, None
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return False, None
        return True, frame

    def set(self, *_args, **_kwargs):
        return False

    def release(self):
        self._opened = False


def _open_protect_snapshot(config):
    protect_id = config.get("protect_id")
    nvr_ip = config.get("nvr_ip") or config.get("ip")
    if not protect_id or not nvr_ip:
        return None, None
    cap = ProtectSnapshotCapture(nvr_ip, protect_id, config["name"])
    ok, frame = cap.read()
    if ok and frame is not None:
        print(f"  Connected using Protect snapshot ({nvr_ip})")
        return cap, config["name"]
    cap.release()
    return None, None


def _camera_key(config):
    return config.get("protect_id") or config.get("id") or config["name"]


def open_camera(config):
    """
    Try a short list of UniFi RTSP URLs, then Protect snapshots.
    Connections run in parallel (up to 4) so NVR 2/3 are not stuck
    behind NVR 1.
    """
    camera_name = config["name"]
    print(f"Connecting to {camera_name} at {config['ip']}...")

    if config.get("online") is False:
        print(f"  {camera_name} is offline in Protect — using snapshot if available")
        cap, name = _open_protect_snapshot(config)
        if cap is not None:
            return cap, name
        return None, None

    cache_key = _camera_key(config)
    brand = (config.get("nvr_brand") or "").lower()
    url_limit = 2 if brand == "unifi" else 4
    rtsp_urls = list(config.get("rtsp_urls") or [])[:url_limit]
    cached_url = _working_rtsp_urls.get(cache_key)
    if cached_url:
        rtsp_urls = [cached_url] + [url for url in rtsp_urls if url != cached_url]

    with _open_camera_sema:
        for rtsp_url in rtsp_urls:
            safe_log_url = rtsp_url.replace(PASS_ENC, "****") if PASS_ENC else rtsp_url
            if "://" in safe_log_url:
                scheme, rest = safe_log_url.split("://", 1)
                host_and_path = rest.split("@")[-1] if "@" in rest else rest
                path = host_and_path.split("/", 1)[1] if "/" in host_and_path else ""
                if path and "PLACEHOLDER" not in path:
                    host = host_and_path.split("/", 1)[0]
                    safe_log_url = f"{scheme}://{host}/***"
            print(f"  Trying: {safe_log_url}")

            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_OPEN_TIMEOUT_MS)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_OPEN_TIMEOUT_MS)

            if not cap.isOpened():
                cap.release()
                continue
            
            ret, frame = cap.read()
            if ret and frame is not None:
                _working_rtsp_urls[cache_key] = rtsp_url
                print(f"  Connected using: {safe_log_url}")
                return cap, camera_name

            cap.release()

    print(f"  RTSP did not work for {camera_name}; trying Protect snapshot...")
    cap, name = _open_protect_snapshot(config)
    if cap is not None:
        return cap, name

    print(f"Failed to connect to {camera_name} after RTSP and snapshot")
    return None, None


def main():

    global running

    threads = []

    print("Starting Camera Threads...")

    for config in CAMERA_CONFIGS:

        t = threading.Thread(
            target=camera_worker,
            args=(config,),
            daemon=True
        )

        t.start()

        threads.append(t)

    print(f"{len(threads)} Camera Thread(s) Started.")

    while True:

        with frame_lock:
            frames = list(latest_frames.values())

        if len(frames) == 0:

            cv2.waitKey(1)
            continue

        resized = []

        for frame in frames:

            resized.append(
                cv2.resize(frame, (640,360))
            )

        if len(resized) == 1:

            display = resized[0]

        elif len(resized) == 2:

            display = cv2.hconcat(resized)

        elif len(resized) == 3:

            blank = resized[0].copy()
            blank[:] = 0

            top = cv2.hconcat(resized[:2])
            bottom = cv2.hconcat([resized[2], blank])

            display = cv2.vconcat([top,bottom])

        else:

            rows=[]

            for i in range(0,len(resized),2):

                if i+1 < len(resized):

                    row=cv2.hconcat([
                        resized[i],
                        resized[i+1]
                    ])

                else:

                    blank=resized[i].copy()
                    blank[:]=0

                    row=cv2.hconcat([
                        resized[i],
                        blank
                    ])

                rows.append(row)

            display=cv2.vconcat(rows)

        cv2.imshow(
            "PPE Detection - Multi Camera",
            display
        )

        key=cv2.waitKey(1)

        if key & 0xFF==ord("q"):

            break

    running=False

    time.sleep(1)

    cv2.destroyAllWindows()

    alarm.stop()

if __name__ == "__main__":
    main()