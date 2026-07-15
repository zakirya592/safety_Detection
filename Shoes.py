import cv2
from ultralytics import YOLO
from alarm import Alarm
from screenshot import ScreenshotManager

alarm = Alarm()
screenshot_manager = ScreenshotManager()

model = YOLO("bestss.pt")
PPE_CLASSES = model.names
print("Model classes:", PPE_CLASSES)

CLASS_COLORS = {
    "shoes": (0, 255, 0),     # Green
    "no_shoes": (0, 0, 255),  # Red
}
VIOLATION_CLASSES = {"no_shoes"}
TARGET_CLASSES = {"shoes", "no_shoes"}  # Only draw boxes for these

video_path = "https://assets.mixkit.co/videos/23410/23410-360.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"ERROR: Could not open video source: {video_path}")
    exit()
else:
    print(f"Video source opened successfully: {video_path}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video or failed to read frame.")
        break

    results = model(frame, conf=0.15)

    detected_violations = set()
    violating_persons = []
    annotated = frame.copy()

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            raw_label = PPE_CLASSES[class_id]
            label = raw_label.strip().lower().replace("-", "_")

            print(f"RAW detection -> class_id: {class_id}, label: '{raw_label}', conf: {confidence:.2f}")

            # ONLY process shoes and no_shoes
            if label not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = CLASS_COLORS[label]

            if label in VIOLATION_CLASSES:
                detected_violations.add(label)
                violating_persons.append({
                    'label': label,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': confidence
                })

            # Draw box and label
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{label} {confidence:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

    if detected_violations:
        alarm.play()
        print(f"ALARM: Shoe violations detected - {detected_violations}")
        screenshot_path = screenshot_manager.take_screenshot(frame, violating_persons)
        if screenshot_path:
            print(f"Screenshot saved: {screenshot_path}")
    else:
        alarm.stop()

    cv2.imshow("Shoes Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
alarm.stop()