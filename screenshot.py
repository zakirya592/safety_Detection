import cv2
import os
import numpy as np
from datetime import datetime

from cloudinary_uploader import CloudinaryUploader


class ScreenshotManager:
    """
    Violation screenshots: ZOOM first (3x optical), then FULL FRAME screenshot.
    No digital crop of the person box.
    """

    def __init__(
        self,
        screenshots_dir="screenshots",
        reset_time_seconds=30,
        upload_to_cloudinary=True,
        cloudinary_uploader=None,
        jpeg_quality=98,
        # 100% ≈ full optical zoom (3x on UniFi cameras that support it)
        optical_zoom_percent=100,
        optical_zoom_settle_seconds=2.5,
    ):
        self.screenshots_dir = screenshots_dir
        self.reset_time_seconds = reset_time_seconds
        self.photographed_persons = []
        self.jpeg_quality = jpeg_quality
        self.optical_zoom_percent = optical_zoom_percent
        self.optical_zoom_settle_seconds = optical_zoom_settle_seconds
        self.cloudinary_uploader = (
            cloudinary_uploader
            if cloudinary_uploader is not None
            else (CloudinaryUploader() if upload_to_cloudinary else None)
        )

        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

    def _clean_old_photographed_persons(self):
        current_time = datetime.now()
        self.photographed_persons = [
            person for person in self.photographed_persons
            if (current_time - person["timestamp"]).total_seconds() < self.reset_time_seconds
        ]

    def _alarm_parts(self, label):
        text = str(label or "").replace("|", ",").replace(" + ", ",")
        return [p.strip() for p in text.split(",") if p.strip()]

    def _draw_full_frame_labels(self, frame, persons, banner_prefix="PPE"):
        """Annotate the FULL frame (no crop). Boxes only when coords match this frame."""
        out = frame.copy()
        labels = []
        for person_info in persons:
            shown = (
                str(person_info["label"])
                .replace("NO-Hardhat", "No Helmet")
                .replace("Hardhat", "Helmet")
            )
            conf = float(person_info.get("confidence") or 0)
            labels.append(f"{shown} {conf:.2f}")

            # Draw box only if it fits this frame (live fallback). After optical
            # zoom, wide-view boxes no longer match — skip drawing them.
            try:
                x1 = int(person_info["x1"])
                y1 = int(person_info["y1"])
                x2 = int(person_info["x2"])
                y2 = int(person_info["y2"])
            except (KeyError, TypeError, ValueError):
                continue
            h, w = out.shape[:2]
            if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    out,
                    f"{shown} {conf:.2f}",
                    (max(5, x1), max(28, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        text = " | ".join(labels) if labels else "PPE Violation"
        cv2.rectangle(out, (0, 0), (out.shape[1], 48), (0, 0, 0), -1)
        cv2.putText(
            out,
            f"{banner_prefix} | {text}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return out

    def _try_optical_zoom_full_screenshot(self, persons, nvr_ip, protect_id):
        """
        1) Set 3x optical zoom
        2) Wait for lens
        3) Full-frame Protect snapshot (NO crop)
        4) Restore zoom
        """
        try:
            from unifi_discover import capture_optically_zoomed_snapshot
        except ImportError:
            return None

        jpeg = capture_optically_zoomed_snapshot(
            nvr_ip,
            protect_id,
            zoom_percent=self.optical_zoom_percent,
            settle_seconds=self.optical_zoom_settle_seconds,
            restore_zoom=0,
        )
        if not jpeg:
            return None
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        print(
            f"Optical zoom FULL screenshot OK ({frame.shape[1]}x{frame.shape[0]}) "
            f"zoom={self.optical_zoom_percent}%"
        )
        # Full zoomed frame only — do not crop; boxes from wide view do not map
        out = frame.copy()
        labels = []
        for person_info in persons:
            shown = (
                str(person_info["label"])
                .replace("NO-Hardhat", "No Helmet")
                .replace("Hardhat", "Helmet")
            )
            labels.append(f"{shown} {float(person_info.get('confidence') or 0):.2f}")
        text = " | ".join(labels) if labels else "PPE Violation"
        cv2.rectangle(out, (0, 0), (out.shape[1], 48), (0, 0, 0), -1)
        cv2.putText(
            out,
            f"3x Optical | {text}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return out

    def take_screenshot(
        self,
        frame,
        violating_persons,
        camera_name=None,
        nvr_ip=None,
        protect_id=None,
        use_optical_zoom=True,
    ):
        """
        Zoom in (3x optical) → full-frame screenshot. No person crop.

        Fallback (no protect_id / zoom fail): full live frame with boxes, still no crop.
        """
        self._clean_old_photographed_persons()

        new_persons_to_photograph = []

        alarm_labels = {
            "NO-Hardhat", "NO-Safety Vest", "no_glove", "no_goggles",
            "no_helmet", "no_goggle", "no_gloves", "no_boots",
            "No Helmet", "No Gloves", "No Goggles",
        }

        for person_info in violating_persons:
            parts = self._alarm_parts(person_info.get("label", ""))
            if not any(
                part in alarm_labels or part.split()[0] in alarm_labels
                for part in parts
            ):
                continue
            is_new_person = True
            for photographed in self.photographed_persons:
                overlap = (
                    (min(person_info["x2"], photographed["x2"]) - max(person_info["x1"], photographed["x1"]))
                    * (min(person_info["y2"], photographed["y2"]) - max(person_info["y1"], photographed["y1"]))
                )
                if overlap > 0:
                    is_new_person = False
                    break

            if is_new_person:
                new_persons_to_photograph.append(person_info)

        if not new_persons_to_photograph:
            return None

        screenshot_frame = None
        zoom_mode = "full"

        # Preferred: lens zoom → full screenshot (not crop)
        if use_optical_zoom and nvr_ip and protect_id:
            screenshot_frame = self._try_optical_zoom_full_screenshot(
                new_persons_to_photograph, nvr_ip, protect_id
            )
            if screenshot_frame is not None:
                zoom_mode = "optical"

        # Fallback: full live frame (still no crop)
        if screenshot_frame is None:
            screenshot_frame = self._draw_full_frame_labels(
                frame, new_persons_to_photograph, banner_prefix="Full frame"
            )
            print(
                "Screenshot: full live frame (optical zoom unavailable — "
                "need protect_id). No crop."
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        violations_str = "_".join(
            str(p["label"]).replace(" ", "_").replace("+", "")
            for p in new_persons_to_photograph
        )
        violations_str = "".join(
            c if c.isalnum() or c in "_-" else "_" for c in violations_str
        )[:80]
        screenshot_path = os.path.join(
            self.screenshots_dir,
            f"violation_{zoom_mode}_{violations_str}_{timestamp}.jpg",
        )
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
                tags=["safety-violation", "ppe-detection", zoom_mode],
                context=context,
            )
            if upload_result:
                image_url = upload_result["url"]
                print(f"Cloudinary upload complete: {image_url}")

        current_time = datetime.now()
        for person_info in new_persons_to_photograph:
            person_with_timestamp = person_info.copy()
            person_with_timestamp["timestamp"] = current_time
            self.photographed_persons.append(person_with_timestamp)

        return {
            "path": screenshot_path,
            "image_url": image_url,
            "persons": new_persons_to_photograph,
            "zoom_mode": zoom_mode,
        }
