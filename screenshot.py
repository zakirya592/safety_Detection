import cv2
import os
from datetime import datetime, timedelta

from cloudinary_uploader import CloudinaryUploader


class ScreenshotManager:
    def __init__(
        self,
        screenshots_dir="screenshots",
        reset_time_seconds=30,
        upload_to_cloudinary=True,
        cloudinary_uploader=None,
    ):
        self.screenshots_dir = screenshots_dir
        self.reset_time_seconds = reset_time_seconds  # Reset photographed persons after this many seconds
        self.photographed_persons = []  # List of dicts with person info and timestamp
        self.cloudinary_uploader = (
            cloudinary_uploader
            if cloudinary_uploader is not None
            else (CloudinaryUploader() if upload_to_cloudinary else None)
        )
        
        # Create screenshots directory
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)
    
    def _clean_old_photographed_persons(self):
        """Remove photographed persons that are older than reset_time_seconds"""
        current_time = datetime.now()
        self.photographed_persons = [
            person for person in self.photographed_persons
            if (current_time - person['timestamp']).total_seconds() < self.reset_time_seconds
        ]
    
    def take_screenshot(self, frame, violating_persons, camera_name=None):
        """
        Take a full screenshot with only violating persons highlighted in red.
        
        Args:
            frame: The original video frame
            violating_persons: List of dictionaries containing violation info (label, x1, y1, x2, y2, confidence)
            camera_name: Optional camera name included in Cloudinary metadata
        
        Returns:
            str: Path to the saved screenshot, or None if no new persons to photograph
        """
        # Clean old photographed persons (allow re-photographing after reset_time_seconds)
        self._clean_old_photographed_persons()
        
        # Check if there are any new persons to photograph
        new_persons_to_photograph = []
        
        alarm_labels = {
            "NO-Hardhat", "NO-Safety Vest", "no_glove", "no_goggles",
            "no_helmet", "no_goggle", "no_gloves", "no_boots",
        }

        for person_info in violating_persons:
            parts = [p.strip() for p in str(person_info.get("label", "")).replace("|", ",").split(",")]
            # Hardhat / helmet means the helmet is on. Do not save that as an alarm.
            if not any(part in alarm_labels for part in parts):
                continue
            # Check if this person was already photographed (similar location)
            is_new_person = True
            for photographed in self.photographed_persons:
                # Check if boxes overlap significantly (same person)
                overlap = (min(person_info['x2'], photographed['x2']) - max(person_info['x1'], photographed['x1'])) * \
                         (min(person_info['y2'], photographed['y2']) - max(person_info['y1'], photographed['y1']))
                if overlap > 0:  # If there's any overlap, consider it the same person
                    is_new_person = False
                    break
            
            if is_new_person:
                new_persons_to_photograph.append(person_info)
        
        if not new_persons_to_photograph:
            return None
        
        # Create a copy of the frame for the screenshot
        screenshot_frame = frame.copy()
        
        # Draw red boxes and labels only for violating persons
        for person_info in new_persons_to_photograph:
            cv2.rectangle(screenshot_frame, (person_info['x1'], person_info['y1']), 
                        (person_info['x2'], person_info['y2']), (0, 0, 255), 2)
            cv2.putText(screenshot_frame, f"{person_info['label']} {person_info['confidence']:.2f}", 
                       (person_info['x1'], person_info['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 0, 255), 2)
        
        # Save screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        violations_str = "_".join([p['label'] for p in new_persons_to_photograph])
        screenshot_path = os.path.join(self.screenshots_dir, f"violation_{violations_str}_{timestamp}.jpg")
        cv2.imwrite(screenshot_path, screenshot_frame)

        image_url = None
        if self.cloudinary_uploader and self.cloudinary_uploader.enabled:
            violation_labels = [p["label"] for p in new_persons_to_photograph]
            context = CloudinaryUploader.build_context(
                camera_name=camera_name,
                violations=violation_labels,
            )
            upload_result = self.cloudinary_uploader.upload(
                screenshot_path,
                tags=["safety-violation", "ppe-detection"],
                context=context,
            )
            if upload_result:
                image_url = upload_result["url"]
                print(f"Cloudinary upload complete: {image_url}")

        # Mark these persons as photographed with current timestamp
        current_time = datetime.now()
        for person_info in new_persons_to_photograph:
            person_with_timestamp = person_info.copy()
            person_with_timestamp['timestamp'] = current_time
            self.photographed_persons.append(person_with_timestamp)

        return {
            "path": screenshot_path,
            "image_url": image_url,
            "persons": new_persons_to_photograph,
        }
