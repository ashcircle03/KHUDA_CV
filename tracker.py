"""BoT-SORT + OSNet 추적, Gallery 이벤트 호출.

ROI 파일 형식 (rois.json):
{
  "A": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
  "B": [[x1,y1], ...]
}
폴리곤 꼭짓점을 시계방향 또는 반시계방향으로 기술한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from detector import DetectionResult, Box, belonging_meta
from gallery import Gallery
import snapshot_store

# BoT-SORT track_buffer 기본값(30)보다 크게 설정해 refind과 충돌 방지
_LOST_PATIENCE = 45  # frames; should be >= BoT-SORT track_buffer



@dataclass
class _TrackState:
    last_seat: Optional[str]


class Tracker:
    def __init__(
        self,
        gallery:      Gallery,
        reid_model:   str = "osnet_x0_25_msmt17.pt",
        roi_path:     str = "rois.json",
        device:       str = "cpu",
    ) -> None:
        self._gallery     = gallery
        self._seat_rois   = _load_rois(roi_path)
        self._tracker     = _build_botsort(reid_model, device)

        # tracklet 생애주기
        self._seen_ids:    set[int]        = set()
        self._lost_cands:  dict[int, int]  = {}   # track_id → 연속 부재 프레임 수
        self._emitted_lost: set[int]       = set()
        self._pending_new:  set[int]       = set()  # gallery 등록 대기 (seat/emb 미확보)

        # track별 상태
        self._track_state: dict[int, _TrackState] = {}

        # gallery.on_lost_tracklet(has_luggage=True) 후 짐 감시 대상 좌석
        self._away_seats: set[str] = set()
        self._last_raw_tracks = []

    # ── 외부 인터페이스 ───────────────────────────────────────────────────

    def update(self, frame: np.ndarray, detections: DetectionResult) -> None:
        """매 프레임 main.py가 호출."""
        raw_tracks = self._run_botsort(frame, detections.person_boxes)
        self._last_raw_tracks = raw_tracks
        active_ids: set[int] = set()

        # ── 1. 활성 track 처리 ──────────────────────────────────────────
        for row in raw_tracks:
            tid  = int(row[4])
            xyxy = row[:4]
            active_ids.add(tid)

            seat = self._find_seat(xyxy)

            if tid not in self._track_state:
                self._track_state[tid] = _TrackState(last_seat=seat)
            elif seat:
                self._track_state[tid].last_seat = seat

            # ── 2. 신규 tracklet ────────────────────────────────────────
            if tid not in self._seen_ids:
                self._seen_ids.add(tid)
                self._pending_new.add(tid)

            self._lost_cands.pop(tid, None)

        # ── 3. pending_new → seat 확보 시 크롭에서 임베딩 추출 후 등록 ──
        for tid in list(self._pending_new):
            st = self._track_state.get(tid)
            if not (st and st.last_seat):
                continue

            crop = self._crop_person(frame, tid)
            emb  = self._embed_crop(crop) if crop is not None else None
            if emb is None:
                emb = np.zeros(512, dtype=np.float32)

            self._away_seats.discard(st.last_seat)
            self._gallery.on_new_tracklet(tid, emb, st.last_seat)
            self._pending_new.discard(tid)

            pid = self._gallery.get_person_id(tid)
            if pid is not None and crop is not None:
                snapshot_store.save(pid, st.last_seat, crop, frame)

        # ── 4. lost debounce ────────────────────────────────────────────
        for tid in self._seen_ids:
            if tid not in active_ids and tid not in self._emitted_lost:
                self._lost_cands[tid] = self._lost_cands.get(tid, 0) + 1

        to_emit = [
            tid for tid, cnt in self._lost_cands.items()
            if cnt >= _LOST_PATIENCE
        ]
        for tid in to_emit:
            self._emitted_lost.add(tid)
            del self._lost_cands[tid]
            self._pending_new.discard(tid)  # 등록 전 소멸 — gallery 호출 불필요

            st = self._track_state.get(tid)
            if st and st.last_seat:
                boxes = self._luggage_boxes_in_seat(st.last_seat, detections.luggage_boxes)
                items = _boxes_to_belongings(boxes)
                # LEFT(짐 없음) 시 스냅샷 삭제
                if not items:
                    pid = self._gallery.get_person_id(tid)
                    if pid is not None:
                        snapshot_store.remove(pid)
                self._gallery.on_lost_tracklet(tid, items)
                if items:
                    self._away_seats.add(st.last_seat)

        # ── 5. AWAY 좌석 짐 소멸 감지 ───────────────────────────────────
        for seat_id in list(self._away_seats):
            if not self._luggage_in_seat(seat_id, detections.luggage_boxes):
                self._away_seats.discard(seat_id)
                self._gallery.on_luggage_lost(seat_id)

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────

    def _run_botsort(
        self, frame: np.ndarray, person_boxes: list[Box]
    ) -> np.ndarray:
        if person_boxes:
            dets = np.array(
                [[*b.xyxy, b.confidence, 0.0] for b in person_boxes],
                dtype=np.float32,
            )
        else:
            dets = np.empty((0, 6), dtype=np.float32)
        result = self._tracker.update(dets, frame)
        return result if result is not None and len(result) > 0 else []

    def _embed_crop(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """크롭 이미지에서 직접 OSNet 임베딩 추출."""
        try:
            reid = getattr(self._tracker, "model", None)
            if reid is None:
                return None
            resized = cv2.resize(crop, (128, 256))      # OSNet 입력 크기
            batch   = resized[np.newaxis]               # (1, H, W, C)
            feat    = reid(batch)                       # (1, D)
            vec     = np.asarray(feat[0], dtype=np.float32)
            norm    = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        except Exception:
            return None

    def _find_seat(self, xyxy: np.ndarray, overlap_thresh: float = 0.15) -> Optional[str]:
        """사람 bbox와 ROI의 겹침 비율이 가장 높은 좌석 반환.

        단일 점 판별 대신 면적 겹침 비율을 써서 앵글·왜곡에 강인하게 대응.
        overlap_thresh: 사람 bbox 면적 중 ROI와 겹치는 비율 최솟값.
        """
        px1, py1, px2, py2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
        p_area = max((px2 - px1) * (py2 - py1), 1e-6)

        best_seat, best_ratio = None, overlap_thresh
        for seat_id, polygon in self._seat_rois.items():
            # 폴리곤 → 바운딩 박스
            pts = polygon.reshape(-1, 2)
            rx1, ry1 = float(pts[:, 0].min()), float(pts[:, 1].min())
            rx2, ry2 = float(pts[:, 0].max()), float(pts[:, 1].max())

            # 교집합 면적
            ix1, iy1 = max(px1, rx1), max(py1, ry1)
            ix2, iy2 = min(px2, rx2), min(py2, ry2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

            ratio = inter / p_area
            if ratio > best_ratio:
                best_ratio, best_seat = ratio, seat_id

        return best_seat

    def _crop_person(self, frame: np.ndarray, tid: int) -> np.ndarray | None:
        """현재 프레임에서 해당 track의 사람 크롭 반환."""
        st = self._track_state.get(tid)
        if st is None:
            return None
        for row in self._last_raw_tracks:
            if int(row[4]) == tid:
                x1,y1,x2,y2 = map(int, row[:4])
                pad = 10
                h, w = frame.shape[:2]
                x1,y1 = max(0,x1-pad), max(0,y1-pad)
                x2,y2 = min(w,x2+pad), min(h,y2+pad)
                return frame[y1:y2, x1:x2]
        return None

    def _luggage_boxes_in_seat(
        self, seat_id: str, luggage_boxes: list[Box]
    ) -> list[Box]:
        """짐 bbox의 중심점이 좌석 ROI 폴리곤 안에 있는 것만 반환."""
        polygon = self._seat_rois.get(seat_id)
        if polygon is None:
            return []
        result = []
        for box in luggage_boxes:
            cx = float((box.xyxy[0] + box.xyxy[2]) / 2)
            cy = float((box.xyxy[1] + box.xyxy[3]) / 2)
            if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                result.append(box)
        return result

    def _luggage_in_seat(self, seat_id: str, luggage_boxes: list[Box]) -> bool:
        return bool(self._luggage_boxes_in_seat(seat_id, luggage_boxes))


# ── 모듈 레벨 팩토리 ─────────────────────────────────────────────────────

def _boxes_to_belongings(boxes: list[Box]) -> list[dict]:
    return [
        {**belonging_meta(b.cls_name), "confidence": round(float(b.confidence), 2)}
        for b in boxes
    ]


def _load_rois(path: str) -> dict[str, np.ndarray]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        data = json.load(f)
    # 각 좌석: [[x,y], ...] → (N,1,2) int32 (pointPolygonTest 요구 형식)
    return {
        k: np.array(v, dtype=np.int32).reshape(-1, 1, 2)
        for k, v in data.items()
    }


def _build_botsort(reid_weights: str, device: str):
    try:
        from boxmot.trackers.tracker_zoo import create_tracker, get_tracker_config
    except ImportError as e:
        raise ImportError("pip install boxmot>=19.0.0") from e

    return create_tracker(
        tracker_type      ="botsort",
        tracker_config    =get_tracker_config("botsort"),
        reid_weights      =Path(reid_weights),
        device            =device,
        half              =False,
        per_class         =False,
    )
