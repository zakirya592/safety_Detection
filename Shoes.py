import cv2
from ultralytics import YOLO
from alarm import Alarm
from screenshot import ScreenshotManager

# Initialize alarm
alarm = Alarm()

# Initialize screenshot manager
screenshot_manager = ScreenshotManager()

# Load both YOLO models
boots_model = YOLO("best11.pt")
ppe_model = YOLO("best.pt")

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

# Only these get drawn from the boots model
BOOTS_SHOW_LABELS = {"boots", "no_boots"}
BOOTS_VIOLATION_LABELS = {"no_boots"}

# Only these get drawn from the PPE model
PPE_SHOW_LABELS = {"Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest", "Person"}
PPE_VIOLATION_LABELS = {"NO-Hardhat", "NO-Safety Vest"}

# Video path
video_path = "https://www.shutterstock.com/shutterstock/videos/1099083587/preview/stock-footage-teamwork-of-black-workers-working-in-large-warehouse-store-industry-rack-of-stock-storage-interior.webm"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    # Track which violations are detected (combined from both models)
    detected_violations = set()

    # Track violating persons for screenshots (combined from both models)
    violating_persons = []

    # Create annotated frame
    annotated = frame.copy()

    # ---- Run boots model ----
    boots_results = boots_model(frame)
    for result in boots_results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = BOOTS_CLASSES.get(class_id, str(class_id))

            if label not in BOOTS_SHOW_LABELS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label in BOOTS_VIOLATION_LABELS:
                color = (0, 0, 255)  # Red for no_boots
                detected_violations.add(label)
                violating_persons.append({
                    'label': label,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': confidence
                })
            else:
                color = (0, 255, 0)  # Green for boots

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{label} {confidence:.2f}",
                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, color, 2)

    # ---- Run PPE model (Hardhat/Vest/Person) ----
    ppe_results = ppe_model(frame)
    for result in ppe_results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = PPE_CLASSES.get(class_id, str(class_id))

            if label not in PPE_SHOW_LABELS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label in PPE_VIOLATION_LABELS:
                color = (0, 0, 255)  # Red for violations
                detected_violations.add(label)
                violating_persons.append({
                    'label': label,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': confidence
                })
            else:
                color = (0, 255, 0)  # Green for compliant PPE / Person

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{label} {confidence:.2f}",
                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, color, 2)

    # Play alarm if ANY violation from EITHER model is detected
    if detected_violations:
        alarm.play()
        print(f"ALARM: PPE violations detected - {detected_violations}")

        screenshot_path = screenshot_manager.take_screenshot(frame, violating_persons)
        if screenshot_path:
            print(f"Screenshot saved: {screenshot_path}")
    else:
        alarm.stop()

    # Display annotated frame
    cv2.imshow("PPE Detection", annotated)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
alarm.stop()