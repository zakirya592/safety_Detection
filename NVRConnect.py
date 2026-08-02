import cv2
from urllib.parse import quote
from ultralytics import YOLO
from alarm import Alarm
from screenshot import ScreenshotManager
from detection_alert_db import save_detection_alerts_async
import os
import threading
from dotenv import load_dotenv

# Initialize alarm
alarm = Alarm()
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

RTSP_OPEN_TIMEOUT_MS = 5000
_working_rtsp_urls = {}
_open_camera_lock = threading.Lock()

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
HELMET_POSITIVE = {"Hardhat"}
HELMET_NEGATIVE = {"NO-Hardhat"}
VEST_POSITIVE = {"Safety Vest"}
VEST_NEGATIVE = {"NO-Safety Vest"}
GLOVE_POSITIVE = {"glove"}
GLOVE_NEGATIVE = {"no_glove"}

GOGGLES_POSITIVE = {"goggles"}
GOGGLES_NEGATIVE = {"no_goggles"}

# All the item labels we bother drawing/considering (Person is handled separately)
ITEM_LABELS = HELMET_POSITIVE | HELMET_NEGATIVE | VEST_POSITIVE | VEST_NEGATIVE | GLOVE_POSITIVE | GLOVE_NEGATIVE | GOGGLES_POSITIVE | GOGGLES_NEGATIVE


# Confidence threshold for Person class only (lowered to 30% to detect more people)
PERSON_CONFIDENCE_THRESHOLD = 0.30

# Fraction of an item's own box area that must fall inside a person's box
# for that item to be considered "worn by" that person.
ITEM_CONTAINMENT_THRESHOLD = 0.5

# Performance optimization settings
PROCESS_EVERY_N_FRAMES = 40  # Process every Nth frame to improve performance
MODEL_INPUT_SIZE = 192       # Smaller input size for faster inference

# Person tracking settings
MAX_MISSING_FRAMES = 10  # Remove tracked person after 10 consecutive frames without detection
IOU_THRESHOLD = 0.3       # Intersection over Union threshold for matching detections to tracks

RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)


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
# Camera / NVR configuration
# ---------------------------------------------------------------------------
# IMPORTANT: build RTSP URLs with urllib.parse.quote so special characters
# in the username/password (like "@") are always encoded correctly and
# consistently. Hand-typing "%40" in a string is error-prone and, if you
# accidentally also include the raw "@" version, that raw version is
# actually an invalid URL (two "@" symbols confuses the parser: it splits
# on the LAST "@", so the password and part of the host get merged into
# garbage). Never include the un-encoded form as a fallback.

RAW_USERNAME = os.environ.get("NVR_USER_NAME")
RAW_PASSWORD = os.environ.get("RAW_PASSWORD")
NVR_IP = os.environ.get("NVR_IP")
RTSP_PORT = os.environ.get("RTSP_PORT")

USER_ENC = quote(RAW_USERNAME, safe="")
PASS_ENC = quote(RAW_PASSWORD, safe="")


def build_rtsp_urls(ip, port, channel=1, user_enc=USER_ENC, pass_enc=PASS_ENC):
    """
    Build a list of RTSP URL candidates covering the common NVR/camera
    brand conventions (Hikvision-style, Dahua-style, Uniview-style).
    All candidates use properly percent-encoded credentials.
    """
    auth = f"{user_enc}:{pass_enc}@{ip}:{port}"
    hik_channel = f"{channel}01"  # e.g. channel 1 -> 101, channel 2 -> 201

    return [
        # Uniview-style (IPC2122LB cameras / Uniview NVR — try first)
        f"rtsp://{auth}/unicast/c{channel}/s0/live",
        f"rtsp://{auth}/unicast/c{channel}/s1/live",
        f"rtsp://{auth}/media/video{channel}",

        # Hikvision-style
        f"rtsp://{auth}/Streaming/Channels/{hik_channel}",
        f"rtsp://{auth}/Streaming/Channels/{hik_channel}/main",
        f"rtsp://{auth}/Streaming/Channels/{channel}02",  # sub stream

        # Dahua-style
        f"rtsp://{auth}/cam/realmonitor?channel={channel}&subtype=0",
        f"rtsp://{auth}/cam/realmonitor?channel={channel}&subtype=1",

        # Generic fallbacks
        f"rtsp://{auth}/channel{channel}",
        f"rtsp://{auth}/stream{channel}",
    ]


# List every channel number the NVR has a camera attached to (from your
# NVR's Camera Management list this was D1 and D2, i.e. channels 1 and 2).
# Add more numbers here if you connect additional cameras later.
ACTIVE_CHANNELS = {
    1: {'location': 'Production Line'},
    2: {'location': 'Warehouse Entrance'},
}

CAMERA_CONFIGS = [
    {
        'name': f'NVR Channel {channel}',
        'location': info.get('location', 'Unknown'),
        'rtsp_urls': build_rtsp_urls(NVR_IP, RTSP_PORT, channel=channel),
        'ip': NVR_IP
    }
    for channel, info in ACTIVE_CHANNELS.items()
]

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
      1. Detect every person (from both models).
      2. Detect every PPE item (Hardhat/no_helmet, vest/no_vest, etc.).
      3. For each person, decide Hardhat/vest status from the items that
         fall inside that person's box.
      4. Draw ONE box per person: green + dimensions while compliant/
         unknown, red + "NO Hardhat / NO Vest" label the moment either
         item is confirmed missing.
    """
    violating_persons = []
    annotated = frame.copy()

    if frame_count % PROCESS_EVERY_N_FRAMES != 0:
        active_tracks = person_tracker.update([])
        for track in active_tracks:
            draw_person_box(annotated, track['box'], track['status'],
                             track['confidence'], track['track_id'])

        cv2.putText(annotated, camera_name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return annotated

    # ---- Run boots model with smaller input size ----
    boots_results = boots_model(frame, imgsz=MODEL_INPUT_SIZE, verbose=False)
    raw_detections = []

    for result in boots_results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = BOOTS_CLASSES.get(class_id, str(class_id))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_detections.append({'box': [x1, y1, x2, y2], 'label': label, 'confidence': confidence})

    # ---- Run PPE model with smaller input size ----
    ppe_results = ppe_model(frame, imgsz=MODEL_INPUT_SIZE, verbose=False)

    for result in ppe_results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = PPE_CLASSES.get(class_id, str(class_id))

            if label == "Person" and confidence < PERSON_CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_detections.append({'box': [x1, y1, x2, y2], 'label': label, 'confidence': confidence})

    # Split into persons vs PPE items
    person_detections = [d for d in raw_detections if d['label'] == "Person"]
    item_detections = [d for d in raw_detections if d['label'] in ITEM_LABELS]

    # De-duplicate overlapping person boxes coming from the two models
    # (keep the highest-confidence box out of any pair that overlaps a lot)
    deduped_persons = []
    used = set()
    for i, p1 in enumerate(person_detections):
        if i in used:
            continue
        group = [i]
        for j, p2 in enumerate(person_detections):
            if j <= i or j in used:
                continue
            if calculate_iou(p1['box'], p2['box']) > 0.5:
                group.append(j)
        best = max(group, key=lambda idx: person_detections[idx]['confidence'])
        deduped_persons.append(person_detections[best])
        used.update(group)

    detected_persons = []
    for person in deduped_persons:
        status = classify_person_ppe(person['box'], item_detections)
        detected_persons.append({
            'box': person['box'],
            'confidence': person['confidence'],
            'status': status,
        })

        if status["is_violation"]:
            violating_persons.append({
                'label': " & ".join(status["missing_items"]),
                'x1': person['box'][0], 'y1': person['box'][1],
                'x2': person['box'][2], 'y2': person['box'][3],
                'confidence': person['confidence'],
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
        draw_person_box(annotated, track['box'], track['status'],
                         track['confidence'], track['track_id'])

    cv2.putText(annotated, camera_name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return annotated


def open_camera(config):
    """
    Try each candidate RTSP URL in order until one both opens AND
    successfully returns a frame. Returns (cap, name) or (None, None).
    """
    camera_name = config["name"]
    print(f"Connecting to {camera_name} at {config['ip']}...")

    rtsp_urls = list(config["rtsp_urls"])
    cached_url = _working_rtsp_urls.get(camera_name)
    if cached_url:
        rtsp_urls = [cached_url] + [url for url in rtsp_urls if url != cached_url]

    with _open_camera_lock:
        for rtsp_url in rtsp_urls:
            safe_log_url = rtsp_url.replace(PASS_ENC, "****")
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
                _working_rtsp_urls[camera_name] = rtsp_url
                print(f"  Connected using: {safe_log_url}")
                return cap, camera_name

            cap.release()

    print(f"Failed to connect to {camera_name} after trying all URL formats")
    return None, None


def main():
    caps = []
    for config in CAMERA_CONFIGS:
        cap, name = open_camera(config)
        if cap is not None:
            caps.append((cap, name))

    if not caps:
        return

    print(f"Connected to {len(caps)} camera(s). Press 'q' to quit.")

    frame_counters = {name: 0 for _, name in caps}
    person_trackers = {name: PersonTracker() for _, name in caps}

    while True:
        snapshots = []
        for cap, name in caps:
            success, frame = cap.read()
            if success:
                snapshots.append((name, frame))
            else:
                print(f"Failed to read from {name}")

        frames = []
        for name, frame in snapshots:
            frame_counters[name] += 1
            processed_frame = process_frame(
                frame, name, frame_counters[name], person_trackers[name]
            )
            frames.append(processed_frame)

        if frames:
            if len(frames) == 1:
                display_frame = frames[0]
            elif len(frames) == 2:
                frame1 = cv2.resize(frames[0], (640, 360))
                frame2 = cv2.resize(frames[1], (640, 360))
                display_frame = cv2.hconcat([frame1, frame2])
            else:
                display_frame = cv2.vconcat([cv2.hconcat(frames[i:i + 2]) for i in range(0, len(frames), 2)])

            cv2.imshow("PPE Detection - Multi-Camera", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap, name in caps:
        cap.release()
    cv2.destroyAllWindows()
    alarm.stop()


if __name__ == "__main__":
    main()