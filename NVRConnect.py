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
# Detection rules (exact classes the user wants):
#   1) ppe_model  → Person first
#   2) ppe_model  → Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest
#   3) boots_model → glove, goggles, no_glove, no_goggles
#   ONE full-person box. Text above.
#   NO-* / no_* → RED + alarm + screenshot
#   Hardhat / Safety Vest / glove / goggles / Person → GREEN
# ---------------------------------------------------------------------------
PPE_POS_LABELS = {"Hardhat", "Safety Vest"}
PPE_NEG_LABELS = {"NO-Hardhat", "NO-Safety Vest"}
BOOTS_POS_LABELS = {"glove", "goggles"}
BOOTS_NEG_LABELS = {"no_glove", "no_goggles"}

PPE_ITEM_LABELS = PPE_POS_LABELS | PPE_NEG_LABELS
BOOTS_ITEM_LABELS = BOOTS_POS_LABELS | BOOTS_NEG_LABELS
ALL_NEG_LABELS = PPE_NEG_LABELS | BOOTS_NEG_LABELS
ALL_POS_LABELS = PPE_POS_LABELS | BOOTS_POS_LABELS


def _norm_label(label):
    s = str(label).strip().lower()
    s = s.replace("_", "-").replace(" ", "-")
    while "--" in s:
        s = s.replace("--", "-")
    return s


_LABEL_CANON = {
    "hardhat": "Hardhat",
    "safety-vest": "Safety Vest",
    "safetyvest": "Safety Vest",
    "no-hardhat": "NO-Hardhat",
    "nohardhat": "NO-Hardhat",
    "no-safety-vest": "NO-Safety Vest",
    "nosafetyvest": "NO-Safety Vest",
    "glove": "glove",
    "gloves": "glove",
    "goggles": "goggles",
    "goggle": "goggles",
    "no-glove": "no_glove",
    "noglove": "no_glove",
    "no-gloves": "no_glove",
    "no-goggles": "no_goggles",
    "nogoggles": "no_goggles",
    "no-goggle": "no_goggles",
}


def _canon_item_label(label):
    """Map raw model label to one of the allowed display names, or None."""
    return _LABEL_CANON.get(_norm_label(label))


# Confidence threshold for Person class only
PERSON_CONFIDENCE_THRESHOLD = 0.30

ITEM_CONTAINMENT_THRESHOLD = 0.20
PERSON_ASSOC_PAD = 0.45

PROCESS_EVERY_N_FRAMES = 8
PERSON_INPUT_SIZE = 640
PPE_INPUT_SIZE = 640
BOOTS_INPUT_SIZE = 640
PPE_ITEM_CONFIDENCE = 0.20
BOOTS_ITEM_CONFIDENCE = 0.15

MAX_MISSING_FRAMES = 10
IOU_THRESHOLD = 0.3

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


def _class_ids_for_labels(model, wanted_labels):
    """Resolve YOLO class indices for a set of display/raw label names."""
    wanted_norm = {_norm_label(x) for x in wanted_labels}
    ids = []
    for idx, name in _model_names(model).items():
        if _norm_label(name) in wanted_norm or _canon_item_label(name) in wanted_labels:
            ids.append(idx)
    return ids


def detect_persons(frame):
    """Step 1: detect Person from ppe_model only."""
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
    persons = [
        d for d in _collect_detections(results, ppe_model, PPE_CLASSES)
        if _is_person_label(d["label"])
    ]
    return _dedupe_persons(persons)


def detect_ppe_for_persons(frame, person_boxes):
    """
    Step 2: after Person is found, detect gear on each person crop.

    ppe_model  → Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest
    boots_model → glove, goggles, no_glove, no_goggles
    """
    items = []
    seen = set()

    def _add_item(det, allowed):
        canon = _canon_item_label(det["label"])
        if canon is None or canon not in allowed:
            return
        key = (canon, tuple(det["box"]))
        if key in seen:
            return
        seen.add(key)
        items.append({
            "box": det["box"],
            "label": canon,
            "confidence": det["confidence"],
        })

    ppe_ids = _class_ids_for_labels(ppe_model, PPE_ITEM_LABELS)
    boots_ids = _class_ids_for_labels(boots_model, BOOTS_ITEM_LABELS)

    for person_box in person_boxes:
        x1, y1, x2, y2 = _expand_box(person_box, frame.shape, PERSON_ASSOC_PAD)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 32:
            continue

        with _inference_lock:
            crop_ppe_kwargs = {
                "source": crop,
                "imgsz": PPE_INPUT_SIZE,
                "conf": PPE_ITEM_CONFIDENCE,
                "verbose": False,
            }
            if ppe_ids:
                crop_ppe_kwargs["classes"] = ppe_ids
            crop_boots_kwargs = {
                "source": crop,
                "imgsz": BOOTS_INPUT_SIZE,
                "conf": BOOTS_ITEM_CONFIDENCE,
                "verbose": False,
            }
            if boots_ids:
                crop_boots_kwargs["classes"] = boots_ids
            crop_ppe = ppe_model.predict(**crop_ppe_kwargs)
            crop_boots = boots_model.predict(**crop_boots_kwargs)

        for det in _collect_detections(crop_ppe, ppe_model, PPE_CLASSES, min_conf=PPE_ITEM_CONFIDENCE):
            shifted = {**det, "box": _shift_box(det["box"], x1, y1)}
            _add_item(shifted, PPE_ITEM_LABELS)
        for det in _collect_detections(crop_boots, boots_model, BOOTS_CLASSES, min_conf=BOOTS_ITEM_CONFIDENCE):
            shifted = {**det, "box": _shift_box(det["box"], x1, y1)}
            _add_item(shifted, BOOTS_ITEM_LABELS)

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


def _item_belongs_to_person(item_box, person_box):
    cx = (item_box[0] + item_box[2]) / 2.0
    cy = (item_box[1] + item_box[3]) / 2.0
    if person_box[0] <= cx <= person_box[2] and person_box[1] <= cy <= person_box[3]:
        return True
    if calculate_iou(item_box, person_box) >= 0.15:
        return True
    return containment_ratio(item_box, person_box) >= ITEM_CONTAINMENT_THRESHOLD


def classify_person_ppe(person_box, item_detections, frame_shape=None):
    """
    Build ONE status for the person box.

    RED  if any of: NO-Hardhat, NO-Safety Vest, no_glove, no_goggles
    GREEN otherwise, listing Hardhat / Safety Vest / glove / goggles / Person
    """
    if frame_shape is not None:
        assoc_box = _expand_box(person_box, frame_shape, PERSON_ASSOC_PAD)
    else:
        assoc_box = person_box

    # Best confidence per canonical label for this person
    best_conf = {}
    matched_items = []
    for item in item_detections:
        if not _item_belongs_to_person(item["box"], assoc_box):
            continue
        label = item["label"]
        matched_items.append(item)
        conf = float(item.get("confidence") or 0.0)
        if conf > best_conf.get(label, 0.0):
            best_conf[label] = conf

    # Conflict: if both Hardhat and NO-Hardhat, NO wins (same for vest/gloves/goggles)
    pairs = [
        ("Hardhat", "NO-Hardhat"),
        ("Safety Vest", "NO-Safety Vest"),
        ("glove", "no_glove"),
        ("goggles", "no_goggles"),
    ]
    for pos, neg in pairs:
        if neg in best_conf:
            best_conf.pop(pos, None)

    neg_found = [lab for lab in ("NO-Hardhat", "NO-Safety Vest", "no_glove", "no_goggles") if lab in best_conf]
    pos_found = [lab for lab in ("Hardhat", "Safety Vest", "glove", "goggles") if lab in best_conf]

    is_violation = len(neg_found) > 0
    missing_items = list(neg_found)

    if is_violation:
        label_text = ", ".join(neg_found)
    else:
        # Always include Person; add positive gear that was seen
        parts = ["Person"] + pos_found
        label_text = " | ".join(parts)

    return {
        "Hardhat": "missing" if "NO-Hardhat" in best_conf else ("present" if "Hardhat" in best_conf else "unknown"),
        "vest": "missing" if "NO-Safety Vest" in best_conf else ("present" if "Safety Vest" in best_conf else "unknown"),
        "gloves": "missing" if "no_glove" in best_conf else ("present" if "glove" in best_conf else "unknown"),
        "goggles": "missing" if "no_goggles" in best_conf else ("present" if "goggles" in best_conf else "unknown"),
        "matched_items": matched_items,
        "missing_items": missing_items,
        "is_violation": is_violation,
        "is_fully_compliant": (not is_violation) and ("Hardhat" in best_conf) and ("Safety Vest" in best_conf),
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
    ONE full-person box only:
      RED  + text above  → NO-Hardhat / NO-Safety Vest / no_glove / no_goggles
      GREEN + text above → Person / Hardhat / Safety Vest / glove / goggles
    """
    x1, y1, x2, y2 = box
    color = RED if status.get("is_violation") else GREEN

    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)

    label = status.get("label") or ("Person" if not status.get("is_violation") else "NO-PPE")
    id_part = f"ID{track_id} " if track_id is not None else ""
    text = f"{id_part}{label}"

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ty = max(th + 8, y1 - 8)
    cv2.rectangle(
        annotated,
        (x1, ty - th - 6),
        (x1 + tw + 8, ty + baseline),
        color,
        -1,
    )
    cv2.putText(
        annotated,
        text,
        (x1 + 4, ty - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    return x2 - x1, y2 - y1


def process_frame(frame, camera_name, frame_count, person_tracker):
    """
    1. Detect Person (ppe_model)
    2. Detect Hardhat/NO-Hardhat/Safety Vest/NO-Safety Vest (ppe_model)
       and glove/goggles/no_glove/no_goggles (boots_model) on that person
    3. ONE full-person box + text above
    4. Alarm + screenshot only when any NO-* / no_* is present
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

    # Step 1: Person
    person_detections = detect_persons(frame)

    if not person_detections:
        alarm.stop()
        person_tracker.update([])
        cv2.putText(annotated, camera_name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated, "No person — waiting", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2)
        return annotated

    # Step 2: PPE / gloves on that person
    item_detections = detect_ppe_for_persons(
        frame, [person["box"] for person in person_detections]
    )
    print(
        f"[{camera_name}] persons={len(person_detections)} "
        f"items={[(d['label'], round(d['confidence'], 2)) for d in item_detections]}"
    )

    detected_persons = []
    for person in person_detections:
        status = classify_person_ppe(
            person["box"], item_detections, frame_shape=frame.shape
        )
        detected_persons.append({
            "box": person["box"],
            "confidence": person["confidence"],
            "status": status,
        })

        if status["is_violation"]:
            violating_persons.append({
                "label": status["label"],
                "x1": person["box"][0], "y1": person["box"][1],
                "x2": person["box"][2], "y2": person["box"][3],
                "confidence": person["confidence"],
            })

    # Alarm + screenshot only for NO-* / no_*
    if violating_persons:
        alarm.play()
        print(f"Violations: {[p['label'] for p in violating_persons]}")
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
            print("Screenshot not saved (cooldown)")
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
        f"Persons: {len(detected_persons)} | Items: {len(item_detections)}",
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