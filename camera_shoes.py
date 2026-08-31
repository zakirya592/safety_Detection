"""
Legacy desktop viewer. Camera IPs and passwords now come from .env /
unifi_cameras.json (the three production UniFi NVRs), not the old
192.168.100.x lab cameras.
"""

from NVRConnect import main

if __name__ == "__main__":
    print("Using production UniFi NVRs from .env (10.10.30.2 / .3 / .4).")
    print("Do not use camera_shoes.py and live_detection_api.py at the same time.")
    print("Prefer:  python unifi_discover.py  then  python live_detection_api.py")
    main()
