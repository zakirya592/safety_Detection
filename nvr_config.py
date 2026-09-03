"""
Multi-NVR configuration for UniFi Protect (production).

IPs and camera counts:
  NVR 1  https://10.10.30.2  — 19 cameras
  NVR 2  https://10.10.30.3  — 20 cameras
  NVR 3  https://10.10.30.4  — 18 cameras

UniFi Protect RTSP is token-based on port 7447.
Run `python unifi_discover.py` on the site PC to log into each NVR,
enable RTSP, and write unifi_cameras.json. This module loads that file
when present; otherwise it builds the expected camera slots from .env.
"""

import os
from urllib.parse import quote

from dotenv import load_dotenv

load_dotenv()

from unifi_discover import CACHE_PATH, NVR_HOSTS, load_cache

RAW_USERNAME = os.environ.get("NVR_USERNAME", "admin")
RAW_PASSWORD = os.environ.get("NVR_PASSWORD", "")
RTSP_QUALITY = int(os.environ.get("UNIFI_RTSP_QUALITY", "2"))

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


def build_unifi_rtsp_urls(ip, rtsp_token, port=7447, stream_quality=None, camera_ip=None):
    """
    UniFi Protect RTSP on port 7447. Only try the quality suffix we want,
    then high quality. Extra camera-IP URLs are skipped — they hang for
    5–30s each and blocked NVR 2/3 from connecting.
    """
    if stream_quality is None:
        stream_quality = RTSP_QUALITY

    token = (rtsp_token or "").strip()
    if not token:
        return []

    base = token[:-2] if token.endswith(("_0", "_1", "_2")) else token
    urls = [
        f"rtsp://{ip}:{port}/{base}_{stream_quality}",
        f"rtsp://{ip}:{port}/{base}_0",
    ]
    seen = []
    for url in urls:
        if url not in seen:
            seen.append(url)
    return seen


def _unique_camera_name(name, cam_def, idx, used_names):
    base = (name or f"Camera {idx}").strip() or f"Camera {idx}"
    mac = (cam_def.get("mac") or "").replace(":", "").replace("-", "")[-6:]
    token = (cam_def.get("rtsp_token") or "")[-4:]
    suffix = mac or token or f"{idx:02d}"
    generic = base.lower() in {
        "g6 bullet",
        "g5 bullet",
        "g4 bullet",
        "g4 dome",
        "g5 dome",
        "g5 ptz",
        "ai 360",
        "camera",
    }
    candidate = f"{base} {suffix}" if generic else base
    if candidate in used_names:
        candidate = f"{base} {suffix}"
    n = 2
    original = candidate
    while candidate in used_names:
        candidate = f"{original} ({n})"
        n += 1
    used_names.add(candidate)
    return candidate


def _placeholder_cameras(count, nvr_name):
    return [
        {
            "name": f"{nvr_name} Cam {i:02d}",
            "location": f"{nvr_name} — Area {i:02d}",
            "rtsp_token": "",
            "camera_ip": "",
        }
        for i in range(1, count + 1)
    ]


def _nvr_from_host(host, discovered=None):
    nvr = {
        "id": host["id"],
        "name": host["name"],
        "ip": host["ip"],
        "brand": "unifi",
        "rtsp_port": 7447,
    }

    if discovered and discovered.get("cameras"):
        nvr["cameras"] = discovered["cameras"]
        return nvr

    nvr["cameras"] = _placeholder_cameras(host["expected_count"], host["name"])
    return nvr


def load_nvr_configs():
    cache = load_cache(CACHE_PATH)
    discovered_by_id = {}
    if cache:
        for nvr in cache.get("nvrs") or []:
            discovered_by_id[nvr.get("id")] = nvr

    return [_nvr_from_host(host, discovered_by_id.get(host["id"])) for host in NVR_HOSTS]


NVR_CONFIGS = load_nvr_configs()


def _rtsp_urls_for_camera(nvr, camera_def, channel_num):
    brand = nvr.get("brand", "hikvision").lower()
    ip = nvr["ip"]
    port = nvr.get("rtsp_port", 554)

    if brand == "unifi":
        return build_unifi_rtsp_urls(
            ip,
            camera_def.get("rtsp_token"),
            port=port,
            camera_ip=camera_def.get("camera_ip"),
        )

    return build_hikvision_rtsp_urls(ip, port, channel=channel_num)


def build_all_camera_configs(nvr_configs=None):
    """Flatten all NVRs into a single ordered camera list with global IDs."""
    if nvr_configs is None:
        nvr_configs = NVR_CONFIGS

    cameras = []
    global_id = 1
    used_names = set()

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
            name = _unique_camera_name(
                cam_def.get("name", f"{nvr_name} Cam {idx}"),
                cam_def,
                idx,
                used_names,
            )
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
                    "camera_ip": cam_def.get("camera_ip", ""),
                    "protect_id": cam_def.get("protect_id"),
                    "stream": cam_def.get("stream", "rtsp"),
                    "online": cam_def.get("online", True),
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


CAMERA_CONFIGS = build_all_camera_configs()
NVR_IP = NVR_CONFIGS[0]["ip"] if NVR_CONFIGS else ""
ACTIVE_CHANNELS = {
    cam["channel"]: {"location": cam["location"], "nvr_id": cam["nvr_id"]}
    for cam in CAMERA_CONFIGS
}

_cache_note = "loaded from unifi_cameras.json" if load_cache(CACHE_PATH) else "placeholders until unifi_discover.py runs"
print(
    f"NVR config: {len(NVR_CONFIGS)} NVR(s), {len(CAMERA_CONFIGS)} camera(s) ({_cache_note})"
)
for _nvr in get_nvr_summary():
    print(f"  {_nvr['id']} {_nvr['ip']}: {_nvr['camera_count']} camera(s)")
