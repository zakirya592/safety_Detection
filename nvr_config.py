"""
Multi-NVR configuration for UniFi (and other brands).

UniFi Protect cameras use token-based RTSP URLs (port 7447), not channel numbers.
Enable RTSP per camera in Protect: Device → Settings → Advanced → RTSP.
Copy the token from the generated URL (rtsp://NVR_IP:7447/TOKEN).

For each NVR, define cameras explicitly OR use channel_count to auto-number channels 1..N
(traditional NVR brands). UniFi should use explicit rtsp_token per camera.
"""

import os
from urllib.parse import quote

# Shared credentials (override via environment variables)
RAW_USERNAME = os.environ.get("NVR_USERNAME", "admin")
RAW_PASSWORD = os.environ.get("NVR_PASSWORD", "Eisa@1234")

USER_ENC = quote(RAW_USERNAME, safe="")
PASS_ENC = quote(RAW_PASSWORD, safe="")


def build_hikvision_rtsp_urls(ip, port, channel=1, user_enc=USER_ENC, pass_enc=PASS_ENC):
    """Hikvision / Dahua / Uniview style RTSP URL candidates."""
    auth = f"{user_enc}:{pass_enc}@{ip}:{port}"
    hik_channel = f"{channel}01"

    return [
        f"rtsp://{auth}/Streaming/Channels/{hik_channel}",
        f"rtsp://{auth}/Streaming/Channels/{hik_channel}/main",
        f"rtsp://{auth}/Streaming/Channels/{channel}02",
        f"rtsp://{auth}/cam/realmonitor?channel={channel}&subtype=0",
        f"rtsp://{auth}/cam/realmonitor?channel={channel}&subtype=1",
        f"rtsp://{auth}/unicast/c{channel}/s0/live",
        f"rtsp://{auth}/unicast/c{channel}/s1/live",
        f"rtsp://{auth}/media/video{channel}",
        f"rtsp://{auth}/channel{channel}",
        f"rtsp://{auth}/stream{channel}",
    ]


def build_unifi_rtsp_urls(ip, rtsp_token, port=7447, stream_quality=0):
    """
  UniFi Protect RTSP (no username/password — token auth).
  stream_quality: 0 = high, 1 = medium, 2 = low (appended as _0, _1, _2 if not in token).
    """
    token = rtsp_token.strip()
    if not token.endswith(("_0", "_1", "_2")):
        token = f"{token}_{stream_quality}"
    return [f"rtsp://{ip}:{port}/{token}"]


# ---------------------------------------------------------------------------
# NVR definitions — edit IPs, tokens, and camera counts for your site
# ---------------------------------------------------------------------------
NVR_CONFIGS = [
    {
        "id": "nvr1",
        "name": "UniFi NVR — Site 1",
        "ip": "192.168.100.9",
        "brand": "unifi",
        "rtsp_port": 7447,
        # 10 cameras — replace PLACEHOLDER tokens with values from UniFi Protect UI
        "cameras": [
            {"name": "NVR1 Cam 01", "location": "Site 1 — Area 01", "rtsp_token": "PLACEHOLDER_TOKEN_01"},
            {"name": "NVR1 Cam 02", "location": "Site 1 — Area 02", "rtsp_token": "PLACEHOLDER_TOKEN_02"},
            {"name": "NVR1 Cam 03", "location": "Site 1 — Area 03", "rtsp_token": "PLACEHOLDER_TOKEN_03"},
            {"name": "NVR1 Cam 04", "location": "Site 1 — Area 04", "rtsp_token": "PLACEHOLDER_TOKEN_04"},
            {"name": "NVR1 Cam 05", "location": "Site 1 — Area 05", "rtsp_token": "PLACEHOLDER_TOKEN_05"},
            {"name": "NVR1 Cam 06", "location": "Site 1 — Area 06", "rtsp_token": "PLACEHOLDER_TOKEN_06"},
            {"name": "NVR1 Cam 07", "location": "Site 1 — Area 07", "rtsp_token": "PLACEHOLDER_TOKEN_07"},
            {"name": "NVR1 Cam 08", "location": "Site 1 — Area 08", "rtsp_token": "PLACEHOLDER_TOKEN_08"},
            {"name": "NVR1 Cam 09", "location": "Site 1 — Area 09", "rtsp_token": "PLACEHOLDER_TOKEN_09"},
            {"name": "NVR1 Cam 10", "location": "Site 1 — Area 10", "rtsp_token": "PLACEHOLDER_TOKEN_10"},
        ],
    },
    {
        "id": "nvr2",
        "name": "UniFi NVR — Site 2",
        "ip": "192.168.100.10",
        "brand": "unifi",
        "rtsp_port": 7447,
        # 20 cameras
        "cameras": [
            {
                "name": f"NVR2 Cam {i:02d}",
                "location": f"Site 2 — Area {i:02d}",
                "rtsp_token": f"PLACEHOLDER_TOKEN_{i:02d}",
            }
            for i in range(1, 21)
        ],
    },
    {
        "id": "nvr3",
        "name": "UniFi NVR — Site 3",
        "ip": "192.168.100.11",
        "brand": "unifi",
        "rtsp_port": 7447,
        # 30 cameras
        "cameras": [
            {
                "name": f"NVR3 Cam {i:02d}",
                "location": f"Site 3 — Area {i:02d}",
                "rtsp_token": f"PLACEHOLDER_TOKEN_{i:02d}",
            }
            for i in range(1, 31)
        ],
    },
]


def _rtsp_urls_for_camera(nvr, camera_def, channel_num):
    brand = nvr.get("brand", "hikvision").lower()
    ip = nvr["ip"]
    port = nvr.get("rtsp_port", 554)

    if brand == "unifi":
        token = camera_def.get("rtsp_token")
        if not token:
            raise ValueError(f"UniFi camera on {nvr['id']} needs rtsp_token")
        return build_unifi_rtsp_urls(ip, token, port=port)

    return build_hikvision_rtsp_urls(ip, port, channel=channel_num)


def build_all_camera_configs(nvr_configs=None):
    """Flatten all NVRs into a single ordered camera list with global IDs."""
    if nvr_configs is None:
        nvr_configs = NVR_CONFIGS

    cameras = []
    global_id = 1

    for nvr in nvr_configs:
        nvr_id = nvr["id"]
        nvr_name = nvr.get("name", nvr_id)
        nvr_ip = nvr["ip"]

        if "cameras" in nvr:
            camera_list = nvr["cameras"]
        elif "channels" in nvr:
            camera_list = [
                {
                    "channel": ch,
                    "name": f"{nvr_name} Ch {ch}",
                    "location": info.get("location", "Unknown"),
                }
                for ch, info in nvr["channels"].items()
            ]
        elif "channel_count" in nvr:
            prefix = nvr.get("location_prefix", "Camera")
            camera_list = [
                {
                    "channel": i,
                    "name": f"{nvr_name} Ch {i}",
                    "location": f"{prefix} {i}",
                }
                for i in range(1, nvr["channel_count"] + 1)
            ]
        else:
            continue

        for idx, cam_def in enumerate(camera_list, start=1):
            channel = cam_def.get("channel", idx)
            name = cam_def.get("name", f"{nvr_name} Cam {idx}")
            location = cam_def.get("location", "Unknown")

            cameras.append(
                {
                    "id": global_id,
                    "nvr_id": nvr_id,
                    "nvr_name": nvr_name,
                    "nvr_ip": nvr_ip,
                    "nvr_brand": nvr.get("brand", "hikvision"),
                    "camera_index": idx,
                    "channel": channel,
                    "name": name,
                    "location": location,
                    "ip": nvr_ip,
                    "rtsp_urls": _rtsp_urls_for_camera(nvr, cam_def, channel),
                }
            )
            global_id += 1

    return cameras


def get_nvr_summary(nvr_configs=None):
    if nvr_configs is None:
        nvr_configs = NVR_CONFIGS

    cameras = build_all_camera_configs(nvr_configs)
    by_nvr = {nvr["id"]: [] for nvr in nvr_configs}

    for cam in cameras:
        by_nvr[cam["nvr_id"]].append(cam)

    return [
        {
            "id": nvr["id"],
            "name": nvr.get("name", nvr["id"]),
            "ip": nvr["ip"],
            "brand": nvr.get("brand", "hikvision"),
            "rtsp_port": nvr.get("rtsp_port", 554),
            "camera_count": len(by_nvr.get(nvr["id"], [])),
            "cameras": by_nvr.get(nvr["id"], []),
        }
        for nvr in nvr_configs
    ]


# Legacy exports used by NVRConnect / live_detection_api
CAMERA_CONFIGS = build_all_camera_configs()
NVR_IP = NVR_CONFIGS[0]["ip"] if NVR_CONFIGS else ""
ACTIVE_CHANNELS = {
    cam["channel"]: {"location": cam["location"], "nvr_id": cam["nvr_id"]}
    for cam in CAMERA_CONFIGS
}
