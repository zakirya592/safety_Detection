import cv2
from ultralytics import YOLO
from alarm import Alarm
from screenshot import ScreenshotManager

# Initialize alarm
alarm = Alarm()

# Initialize screenshot manager
screenshot_manager = ScreenshotManager()

# Load YOLO model
model = YOLO("best.pt")

# PPE class mapping
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

# Video path
video_path = "https://www.shutterstock.com/shutterstock/videos/4092636713/preview/stock-footage-engineer-men-and-handshake-for-construction-team-with-collaboration-renovation-or-talking.webm"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        break
    
    # Run detection
    results = model(frame)
    
    # Track which violations are detected
    detected_violations = set()
    
    # Track violating persons for screenshots
    violating_persons = []
    
    # Create annotated frame
    annotated = frame.copy()
    
    # Process results
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = PPE_CLASSES[class_id]
            
            # Skip Machinery, Vehicle, Mask, and NO-Mask classes
            if label in ["Machinery", "Vehicle", "Mask", "NO-Mask"]:
                continue
            
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Define colors
            if label in ["NO-Hardhat", "NO-Safety Vest"]:
                color = (0, 0, 255)  # Red for violations
                detected_violations.add(label)
                # Store violating person info for screenshot
                violating_persons.append({
                    'label': label,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': confidence
                })
            else:
                color = (0, 255, 0)  # Green for compliant PPE
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            cv2.putText(annotated, f"{label} {confidence:.2f}", 
                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
    
    # Play alarm if ANY violation is detected
    if detected_violations:
        alarm.play()
        print(f"ALARM: PPE violations detected - {detected_violations}")
        
        # Take screenshot with all violating persons highlighted
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
