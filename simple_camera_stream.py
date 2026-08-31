"""Live preview using the production UniFi NVRs from .env / unifi_cameras.json."""

import cv2

from nvr_config import CAMERA_CONFIGS
from NVRConnect import open_camera


def main():
    caps = []
    for config in CAMERA_CONFIGS:
        cap, name = open_camera(config)
        if cap is not None:
            caps.append((cap, name))

    if not caps:
        print("No cameras connected. Run python unifi_discover.py first.")
        return

    print(f"Connected to {len(caps)} camera(s). Press 'q' to quit.")

    while True:
        frames = []
        for cap, name in caps:
            success, frame = cap.read()
            if not success or frame is None:
                print(f"Failed to read from {name}")
                continue
            annotated = frame.copy()
            cv2.putText(
                annotated,
                name,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            frames.append(cv2.resize(annotated, (640, 360)))

        if frames:
            if len(frames) == 1:
                display_frame = frames[0]
            else:
                rows = []
                for i in range(0, len(frames), 2):
                    pair = frames[i:i + 2]
                    if len(pair) == 1:
                        pair.append(pair[0])
                    rows.append(cv2.hconcat(pair))
                display_frame = cv2.vconcat(rows)
            cv2.imshow("Live Camera Stream", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap, name in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
