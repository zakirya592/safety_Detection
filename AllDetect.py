import cv2
from urllib.parse import quote
from ultralytics import YOLO
from alarm import Alarm
from screenshot import ScreenshotManager
from detection_alert_db import save_detection_alerts_async
import os

# Initialize alarm
alarm = Alarm()
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

# Initialize screenshot manager with 30-second reset time
screenshot_manager = ScreenshotManager(reset_time_seconds=30)

# Load the boots/PPE model
boots_model = YOLO("best11.pt")

# Print the model's actual class mapping once at startup. If this doesn't
# match BOOTS_CLASSES below (e.g. because the model was retrained and the
# class order shifted), goggles/gloves detections will silently be labeled
# as something else or filtered out entirely. Compare this printed dict
# against BOOTS_CLASSES and fix BOOTS_CLASSES if they disagree.
print(f"boots_model.names: {boots_model.names}")

# Class mapping for the boots model
BOOTS_CLASSES = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "none",
    6: "Person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots"
}

# Show boots, vest, goggles, gloves, and Person from the boots model
# (boots/no_boots excluded per requirements; no PPE model needed anymore
# since this model already covers boots/vest/goggles/gloves)
BOOTS_SHOW_LABELS = {"boots", "no_boots", "goggles", "no_goggle", "gloves", "no_gloves", "Person"}
BOOTS_VIOLATION_LABELS = {"no_boots", "no_goggle", "no_gloves"}

# Confidence threshold for Person class only (lowered to 30% to detect more people)
PERSON_CONFIDENCE_THRESHOLD = 0.30

# Set True temporarily to print every raw boots-model detection (class id,
# label, confidence) before any filtering. Use this to confirm goggles/
# gloves are actually being detected by the model at all, and that the
# class_id -> label mapping in BOOTS_CLASSES matches what the model was
# trained with. Turn back off once goggles/gloves are confirmed working.
DEBUG_RAW_DETECTIONS = False

# Performance optimization settings
PROCESS_EVERY_N_FRAMES = 10  # Process every Nth frame to improve performance
MODEL_INPUT_SIZE = 640       # Smaller input size for faster inference

# Person tracking settings
MAX_MISSING_FRAMES = 10  # Remove tracked person after 10 consecutive frames without detection
IOU_THRESHOLD = 0.3       # Intersection over Union threshold for matching detections to tracks


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


class PersonTracker:
    """Track persons across frames to maintain persistent green boxes"""

    def __init__(self):
        self.tracks = {}  # {track_id: {'box': [x1, y1, x2, y2], 'missing_frames': 0, 'label': str}}
        self.next_id = 0

    def calculate_iou(self, box1, box2):
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

    def update(self, detected_persons):
        """
        Update tracks with new detections
        detected_persons: list of {'box': [x1, y1, x2, y2], 'label': str, 'confidence': float}
        Returns: list of all active tracks with their boxes and labels
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
                self.tracks[best_track_id]['label'] = detection['label']
                self.tracks[best_track_id]['confidence'] = detection['confidence']
                matched_track_ids.add(best_track_id)
            else:
                self.tracks[self.next_id] = {
                    'box': detection_box,
                    'missing_frames': 0,
                    'label': detection['label'],
                    'confidence': detection['confidence']
                }
                matched_track_ids.add(self.next_id)
                self.next_id += 1

        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track['missing_frames'] > MAX_MISSING_FRAMES:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        return [
            {
                'track_id': track_id,
                'box': track['box'],
                'label': track['label'],
                'confidence': track['confidence']
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

RAW_USERNAME = "admin"
RAW_PASSWORD = "Eisa@1234"
NVR_IP = "192.168.100.218"
RTSP_PORT = 554

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
        # Hikvision-style
        f"rtsp://{auth}/Streaming/Channels/{hik_channel}",
        f"rtsp://{auth}/Streaming/Channels/{hik_channel}/main",
        f"rtsp://{auth}/Streaming/Channels/{channel}02",  # sub stream

        # Dahua-style
        f"rtsp://{auth}/cam/realmonitor?channel={channel}&subtype=0",
        f"rtsp://{auth}/cam/realmonitor?channel={channel}&subtype=1",

        # Uniview-style (matches the IPC2122LB cameras / Uniview NVR)
        f"rtsp://{auth}/unicast/c{channel}/s0/live",
        f"rtsp://{auth}/unicast/c{channel}/s1/live",
        f"rtsp://{auth}/media/video{channel}",

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


def process_frame(frame, camera_name, frame_count, person_tracker):
    """Process a single frame with both models"""
    detected_violations = set()
    violating_persons = []
    detected_persons = []

    annotated = frame

    if frame_count % PROCESS_EVERY_N_FRAMES != 0:
        active_tracks = person_tracker.update([])
        for track in active_tracks:
            x1, y1, x2, y2 = track['box']
            label = track['label']
            confidence = track['confidence']
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{label} {confidence:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

        cv2.putText(annotated, camera_name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return annotated
        
    small_frame = cv2.resize(frame, (640, 360))
    # ---- Run boots model with smaller input size ----
    boots_results = boots_model( source=small_frame,
    imgsz=640,
    conf=0.25,
    stream=True,
    verbose=False)

    boots_detections = []

    for result in boots_results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = BOOTS_CLASSES.get(class_id, str(class_id))

            if DEBUG_RAW_DETECTIONS:
                print(f"[boots_model raw] class_id={class_id} label={label} conf={confidence:.2f}")

            if label not in BOOTS_SHOW_LABELS:
                continue

            # x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = frame.shape[:2]

            sx = w / 640
            sy = h / 360

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 = int(x1 * sx)
            y1 = int(y1 * sy)
            x2 = int(x2 * sx)
            y2 = int(y2 * sy)

            boots_detections.append({
                'box': [x1, y1, x2, y2],
                'label': label,
                'confidence': confidence
            })

    # Mutual exclusion: goggles/no_goggle, gloves/no_gloves
    filtered_boots_detections = []
    used_boots_indices = set()

    for i, det1 in enumerate(boots_detections):
        if i in used_boots_indices:
            continue

        label1 = det1['label']
        box1 = det1['box']

        conflicting_indices = []
        for j, det2 in enumerate(boots_detections):
            if i == j or j in used_boots_indices:
                continue

            label2 = det2['label']
            box2 = det2['box']

            is_conflicting = (
                (label1 == "goggles" and label2 == "no_goggle") or
                (label1 == "no_goggle" and label2 == "goggles") or
                (label1 == "gloves" and label2 == "no_gloves") or
                (label1 == "no_gloves" and label2 == "gloves") or
                (label1 == "boots" and label2 == "no_boots") or
                (label1 == "no_boots" and label2 == "boots")
            )

            if is_conflicting:
                iou = calculate_iou(box1, box2)
                if iou > 0.3:
                    conflicting_indices.append(j)

        if conflicting_indices:
            all_indices = [i] + conflicting_indices
            best_idx = max(all_indices, key=lambda idx: boots_detections[idx]['confidence'])
            filtered_boots_detections.append(boots_detections[best_idx])
            used_boots_indices.update(all_indices)
        else:
            filtered_boots_detections.append(det1)
            used_boots_indices.add(i)

    for detection in filtered_boots_detections:
        x1, y1, x2, y2 = detection['box']
        label = detection['label']
        confidence = detection['confidence']

        if label == "Person":
            detected_persons.append({
                'box': [x1, y1, x2, y2],
                'label': label,
                'confidence': confidence
            })

        if label in BOOTS_VIOLATION_LABELS:
            color = (0, 0, 255)
            detected_violations.add(label)
            violating_persons.append({
                'label': label,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'confidence': confidence
            })
        else:
            color = (0, 255, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{label} {confidence:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    if detected_violations:
        alarm.play()
        print(f"Violations detected: {detected_violations}")
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
        x1, y1, x2, y2 = track['box']
        label = track['label']
        confidence = track['confidence']
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{label} {confidence:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    cv2.putText(annotated, camera_name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return annotated


def open_camera(config):
    """
    Try each candidate RTSP URL in order until one both opens AND
    successfully returns a frame. Returns (cap, name) or (None, None).
    """
    print(f"Connecting to {config['name']} at {config['ip']}...")

    for rtsp_url in config['rtsp_urls']:
        # Mask password in the printed log line
        safe_log_url = rtsp_url.replace(PASS_ENC, "****")
        print(f"  Trying: {safe_log_url}")

        # cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            cap.release()
            continue

        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"  Connected using: {safe_log_url}")
            return cap, config['name']

        cap.release()

    print(f"Failed to connect to {config['name']} after trying all URL formats")
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
        frames = []
        for cap, name in caps:
            success, frame = cap.read()
            # while cap.grab():
            #     pass
            cap.grab()
            # success, frame = cap.retrieve()
            if success:
                frame_counters[name] += 1
                processed_frame = process_frame(frame, name, frame_counters[name], person_trackers[name])
                frames.append(processed_frame)
            else:
                print(f"Failed to read from {name}")

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