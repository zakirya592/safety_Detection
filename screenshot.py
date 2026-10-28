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
        zoom_pad_ratio=0.40,
        # Cap longest side when cropping; never upscale small boxes (that causes blur).
        max_crop_side=1920,
        jpeg_quality=98,
    ):
        self.screenshots_dir = screenshots_dir
        self.reset_time_seconds = reset_time_seconds  # Reset photographed persons after this many seconds
        self.photographed_persons = []  # List of dicts with person info and timestamp
        self.zoom_pad_ratio = zoom_pad_ratio
        self.max_crop_side = max_crop_side
        self.jpeg_quality = jpeg_quality
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

    def _alarm_parts(self, label):
        text = str(label or "").replace("|", ",").replace(" + ", ",")
        return [p.strip() for p in text.split(",") if p.strip()]

    def _zoom_violation_crop(self, frame, persons):
        """
        Pad + crop violating person(s) at native resolution.
        Only downscale if the crop is huge — never upscale (avoids blur).
        """
        h, w = frame.shape[:2]
        x1 = min(int(p["x1"]) for p in persons)
        y1 = min(int(p["y1"]) for p in persons)
        x2 = max(int(p["x2"]) for p in persons)
        y2 = max(int(p["y2"]) for p in persons)

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = int(bw * self.zoom_pad_ratio)
        pad_y = int(bh * self.zoom_pad_ratio)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return frame.copy()

        ch, cw = crop.shape[:2]
        scale = 1.0
        max_side = max(ch, cw)
        if max_side > self.max_crop_side:
            scale = float(self.max_crop_side) / float(max_side)
            new_w = max(1, int(round(cw * scale)))
            new_h = max(1, int(round(ch * scale)))
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Remap person boxes into crop coords and draw labels
        for person_info in persons:
            px1 = int((int(person_info["x1"]) - x1) * scale)
            py1 = int((int(person_info["y1"]) - y1) * scale)
            px2 = int((int(person_info["x2"]) - x1) * scale)
            py2 = int((int(person_info["y2"]) - y1) * scale)
            cv2.rectangle(crop, (px1, py1), (px2, py2), (0, 0, 255), 2)
            shown = (
                str(person_info["label"])
                .replace("NO-Hardhat", "No Helmet")
                .replace("Hardhat", "Helmet")
            )
            cv2.putText(
                crop,
                f"{shown} {person_info['confidence']:.2f}",
                (max(5, px1), max(25, py1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        return crop

    def take_screenshot(self, frame, violating_persons, camera_name=None):
        """
        Zoom onto the violating person(s), then save that cropped screenshot.

        Args:
            frame: The original video frame
            violating_persons: List of dictionaries containing violation info (label, x1, y1, x2, y2, confidence)
            camera_name: Optional camera name included in Cloudinary metadata

        Returns:
            dict with path/image_url/persons, or None if no new persons to photograph
        """
        # Clean old photographed persons (allow re-photographing after reset_time_seconds)
        self._clean_old_photographed_persons()

        # Check if there are any new persons to photograph
        new_persons_to_photograph = []

        alarm_labels = {
            "NO-Hardhat", "NO-Safety Vest", "no_glove", "no_goggles",
            "no_helmet", "no_goggle", "no_gloves", "no_boots",
            "No Helmet", "No Gloves", "No Goggles",
        }

        for person_info in violating_persons:
            parts = self._alarm_parts(person_info.get("label", ""))
            # Only alarm on missing-gear labels (ignore worn Hardhat / helmet)
            if not any(
                part in alarm_labels or part.split()[0] in alarm_labels
                for part in parts
            ):
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

        # Native-resolution crop (no upscale blur), then high-quality JPEG
        screenshot_frame = self._zoom_violation_crop(frame, new_persons_to_photograph)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        violations_str = "_".join(
            str(p["label"]).replace(" ", "_").replace("+", "")
            for p in new_persons_to_photograph
        )
        # Keep filename short / filesystem-safe
        violations_str = "".join(c if c.isalnum() or c in "_-" else "_" for c in violations_str)[:80]
        screenshot_path = os.path.join(self.screenshots_dir, f"violation_{violations_str}_{timestamp}.jpg")
        cv2.imwrite(
            screenshot_path,
            screenshot_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
        )

        image_url = None
        if self.cloudinary_uploader and self.cloudinary_uploader.enabled:
            violation_labels = [p["label"] for p in new_persons_to_photograph]
            context = CloudinaryUploader.build_context(
                camera_name=camera_name,
                violations=violation_labels,
            )
            upload_result = self.cloudinary_uploader.upload(
                screenshot_path,
                tags=["safety-violation", "ppe-detection", "zoomed"],
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
