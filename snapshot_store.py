"""등록된 사람 스냅샷 저장소 — 좌석당 여러 명 지원.

저장 구조:
  person_id → { thumbnail(크롭), fullImage(풀 프레임), seatId, capturedAt }
"""
from __future__ import annotations

import base64
import threading
from datetime import datetime, timezone, timedelta

import cv2
import numpy as np

KST   = timezone(timedelta(hours=9))
_lock = threading.Lock()
_snapshots: dict[int, dict] = {}   # person_id → snapshot


def _encode(img: np.ndarray, quality: int = 75) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def save(
    person_id:   int,
    seat_id:     str,
    crop:        np.ndarray,   # 사람 크롭 (썸네일용)
    full_frame:  np.ndarray,   # 전체 프레임 (풀 이미지용)
) -> None:
    if crop.size == 0:
        return

    # 풀 프레임은 너비 800px으로 리사이즈
    h, w = full_frame.shape[:2]
    scale = min(1.0, 800 / w)
    full_resized = cv2.resize(full_frame, (int(w*scale), int(h*scale)))

    with _lock:
        _snapshots[person_id] = {
            "personId":    person_id,
            "seatId":      seat_id,
            "thumbnail":   _encode(crop, quality=70),
            "fullImage":   _encode(full_resized, quality=80),
            "capturedAt":  datetime.now(KST).isoformat(),
        }


def remove(person_id: int) -> None:
    with _lock:
        _snapshots.pop(person_id, None)


def get_all() -> list[dict]:
    with _lock:
        return list(_snapshots.values())


def get_by_seat(seat_id: str) -> list[dict]:
    """해당 좌석에 등록된 전원 스냅샷 반환."""
    with _lock:
        return [s for s in _snapshots.values() if s["seatId"] == seat_id]
