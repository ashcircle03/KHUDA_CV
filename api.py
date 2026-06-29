"""FastAPI 서버 — 웹팀 API_SPEC.md 기준 구현.

엔드포인트:
  GET  /api/health
  GET  /api/dashboard
  GET  /api/seats
  GET  /api/seats/layout
  GET  /api/seats/{seatId}
  GET  /api/events
  POST /api/events/{eventId}/action
  GET  /api/settings
  PATCH /api/settings
  GET  /api/cameras/main/stream
  WS   /ws/seats
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from gallery import Gallery
from event_store import EventStore
import snapshot_store

KST = timezone(timedelta(hours=9))


def _now_iso() -> str:
    return datetime.now(KST).isoformat()


# ── 공유 프레임 버퍼 (MJPEG 스트림용) ────────────────────────────────────────

class FrameBuffer:
    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self._frame: Optional[np.ndarray] = None

    def push(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame.copy()

    def get(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame


_frame_buffer = FrameBuffer()


def push_frame(frame: np.ndarray) -> None:
    """main.py가 매 프레임 호출."""
    _frame_buffer.push(frame)


# ── ROI 로딩 ─────────────────────────────────────────────────────────────────

def _load_seat_config(roi_path: str, img_w: int = 1280, img_h: int = 720) -> dict[str, dict]:
    """rois.json → seatId → {seatId, label, roi(정규화 bbox)}"""
    p = Path(roi_path)
    if not p.exists():
        return {}
    with open(p) as f:
        data = json.load(f)
    seats = {}
    for seat_id, polygon in data.items():
        pts = np.array(polygon, dtype=float)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        seats[seat_id] = {
            "seatId": seat_id,
            "label":  seat_id,
            "roi": {
                "x":      round(x1 / img_w, 4),
                "y":      round(y1 / img_h, 4),
                "width":  round((x2 - x1) / img_w, 4),
                "height": round((y2 - y1) / img_h, 4),
            },
        }
    return seats


# ── 추천 문구 ─────────────────────────────────────────────────────────────────

_RECOMMENDATIONS = {
    "OVERDUE":          "추가 주문 또는 좌석 연장 안내가 필요합니다.",
    "NEAR_LIMIT":       "이용 종료 시간이 임박했습니다.",
    "AWAY_TOO_LONG":    "자리비움 시간이 기준을 넘었는지 확인합니다.",
    "BELONGINGS_ONLY":  "물건이 남아 있는지 확인합니다.",
    "NONE":             "",
}

_EVENT_MESSAGES = {
    "SESSION_STARTED": lambda s: f"좌석 {s['seatId']} 이용이 시작되었습니다.",
    "NEAR_LIMIT":      lambda s: f"이용 제한 시간 종료가 임박했습니다.",
    "OVERDUE":         lambda s: f"이용 제한 시간을 초과했습니다.",
    "AWAY_STARTED":    lambda s: f"좌석 {s['seatId']}에서 자리비움이 감지되었습니다.",
    "AWAY_TOO_LONG":   lambda s: f"자리비움 기준 시간을 초과했습니다.",
    "LEFT":            lambda s: f"좌석 {s['seatId']} 이용이 종료되었습니다.",
    "BELONGINGS_ONLY": lambda s: f"사람 없이 물건만 감지되고 있습니다.",
}


# ── Seat 응답 조립 ────────────────────────────────────────────────────────────

def _computed_state(occ: str, alert: str) -> str:
    """프론트엔드 단일 state 필드 (목업 호환)."""
    if occ == "EMPTY":              return "empty"
    if occ == "AWAY":               return "away"
    if alert == "OVERDUE":          return "overdue"
    if alert == "NEAR_LIMIT":       return "near"
    return "seated"


_ALERT_PRIORITY = {"OVERDUE": 4, "NEAR_LIMIT": 3, "AWAY_TOO_LONG": 2, "BELONGINGS_ONLY": 1, "NONE": 0}


def _aggregate_seat(seat_cfg: dict, entries: list[dict]) -> dict:
    """같은 좌석의 여러 gallery entry → 단일 API 응답으로 집계."""
    if not entries:
        return _build_seat(seat_cfg, None)

    # 점유 상태: SEATED 우선
    occs = [e["occupancyState"] for e in entries]
    occ  = "SEATED" if "SEATED" in occs else "AWAY"

    # 가장 심각한 alert
    alert = max(entries, key=lambda e: _ALERT_PRIORITY.get(e["alertState"], 0))["alertState"]

    # 가장 오래 앉은 사람 기준
    max_acc  = max(e["accumulatedSeconds"] for e in entries)
    max_away = max(e["awaySeconds"]        for e in entries)

    # 짐 합산 (중복 제거)
    seen, all_belongings = set(), []
    for e in entries:
        for b in e.get("belongings", []):
            key = b.get("label", "")
            if key not in seen:
                seen.add(key)
                all_belongings.append(b)

    return {
        **seat_cfg,
        "id":                 seat_cfg["seatId"],
        "state":              _computed_state(occ, alert),
        "occupancyState":     occ,
        "alertState":         alert,
        "accumulatedSeconds": max_acc,
        "elapsedSeconds":     max_acc,
        "awaySeconds":        max_away,
        "personCount":        len(entries),
        "hasPerson":          occ == "SEATED",
        "hasBelongings":      bool(all_belongings) or occ == "AWAY",
        "belongings":         all_belongings,
        "confidence":         {"personDetection": 0.0, "belongingsDetection": 0.0, "seatMatch": 0.0},
        "recommendation":     _RECOMMENDATIONS.get(alert, ""),
        "updatedAt":          _now_iso(),
    }


def _build_seat(seat_cfg: dict, gallery_entry: Optional[dict]) -> dict:
    if gallery_entry is None:
        return {
            **seat_cfg,
            "id":                 seat_cfg["seatId"],   # 목업 호환 alias
            "state":              "empty",
            "occupancyState":     "EMPTY",
            "alertState":         "NONE",
            "accumulatedSeconds": 0,
            "elapsedSeconds":     0,                    # 목업 호환 alias
            "awaySeconds":        0,
            "hasPerson":          False,
            "hasBelongings":      False,
            "belongings":         [],
            "confidence":         {"personDetection": 0.0, "belongingsDetection": 0.0, "seatMatch": 0.0},
            "recommendation":     "",
            "updatedAt":          _now_iso(),
        }

    occ   = gallery_entry["occupancyState"]
    alert = gallery_entry["alertState"]
    acc   = gallery_entry["accumulatedSeconds"]

    return {
        **seat_cfg,
        "id":                 seat_cfg["seatId"],
        "state":              _computed_state(occ, alert),
        "occupancyState":     occ,
        "alertState":         alert,
        "accumulatedSeconds": acc,
        "elapsedSeconds":     acc,
        "awaySeconds":        gallery_entry["awaySeconds"],
        "hasPerson":          occ == "SEATED",
        "hasBelongings":      bool(gallery_entry.get("belongings")),
        "belongings":         gallery_entry.get("belongings", []),
        "confidence":         {"personDetection": 0.0, "belongingsDetection": 0.0, "seatMatch": 0.0},
        "recommendation":     _RECOMMENDATIONS.get(alert, ""),
        "updatedAt":          _now_iso(),
    }


def _build_summary(seats: list[dict], unconfirmed: int) -> dict:
    return {
        "totalSeats":        len(seats),
        "seatedSeats":       sum(1 for s in seats if s["occupancyState"] == "SEATED"),
        "awaySeats":         sum(1 for s in seats if s["occupancyState"] == "AWAY"),
        "emptySeats":        sum(1 for s in seats if s["occupancyState"] == "EMPTY"),
        "nearLimitSeats":    sum(1 for s in seats if s["alertState"] == "NEAR_LIMIT"),
        "overdueSeats":      sum(1 for s in seats if s["alertState"] == "OVERDUE"),
        "unconfirmedEvents": unconfirmed,
    }


# ── 진입점 ────────────────────────────────────────────────────────────────────

def start_api(
    gallery:  Gallery,
    roi_path: str = "rois.json",
    host:     str = "0.0.0.0",
    port:     int = 8000,
    img_w:    int = 1280,
    img_h:    int = 720,
) -> None:
    """백그라운드 daemon 스레드로 FastAPI 서버 실행."""
    app          = _build_app(gallery, roi_path, img_w, img_h)
    def _run():
        try:
            uvicorn.run(app, host=host, port=port, log_level="info")
        except Exception as e:
            print(f"[API] 서버 시작 실패: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def _build_app(
    gallery:  Gallery,
    roi_path: str,
    img_w:    int,
    img_h:    int,
) -> FastAPI:
    app          = FastAPI(title="Cafe Seat Monitor")
    events       = EventStore()
    settings     = dict(_DEFAULT_SETTINGS)
    seat_configs = _load_seat_config(roi_path, img_w, img_h)
    ws_clients:  list[WebSocket] = []
    prev_states: dict[str, str]  = {}   # seatId → occupancyState (이벤트 전이 감지용)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────

    def _all_seats() -> list[dict]:
        # 좌석별로 여러 entry 그룹화
        seat_entries: dict[str, list] = {}
        for e in gallery.get_status():
            seat_entries.setdefault(e["seatId"], []).append(e)
        return [
            _aggregate_seat(cfg, seat_entries.get(sid, []))
            for sid, cfg in seat_configs.items()
        ]

    def _detect_events(seats: list[dict]) -> None:
        """상태 전이 감지 → 이벤트 생성."""
        for s in seats:
            sid  = s["seatId"]
            prev = prev_states.get(sid, "EMPTY")
            curr = s["occupancyState"]
            acc  = s["accumulatedSeconds"]
            away = s["awaySeconds"]
            alert = s["alertState"]

            if prev != curr:
                if curr == "SEATED" and prev == "EMPTY":
                    events.add(sid, "SESSION_STARTED", acc, 0,
                               _EVENT_MESSAGES["SESSION_STARTED"](s),
                               "")
                elif curr == "AWAY":
                    events.add(sid, "AWAY_STARTED", acc, 0,
                               _EVENT_MESSAGES["AWAY_STARTED"](s),
                               "")
                elif curr == "EMPTY" and prev in ("SEATED", "AWAY"):
                    events.add(sid, "LEFT", acc, 0,
                               _EVENT_MESSAGES["LEFT"](s),
                               "새 손님에게 안내 가능한 좌석입니다.")

            # 임계 이벤트 (중복 방지는 EventStore 내부에서 처리)
            if alert == "OVERDUE":
                events.add(sid, "OVERDUE", acc, away,
                           _EVENT_MESSAGES["OVERDUE"](s),
                           _RECOMMENDATIONS["OVERDUE"])
            elif alert == "NEAR_LIMIT":
                events.add(sid, "NEAR_LIMIT", acc, away,
                           _EVENT_MESSAGES["NEAR_LIMIT"](s),
                           _RECOMMENDATIONS["NEAR_LIMIT"])
            elif alert == "AWAY_TOO_LONG":
                events.add(sid, "AWAY_TOO_LONG", acc, away,
                           _EVENT_MESSAGES["AWAY_TOO_LONG"](s),
                           _RECOMMENDATIONS["AWAY_TOO_LONG"])

            prev_states[sid] = curr

    # ── REST 엔드포인트 ──────────────────────────────────────────────────

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "serverTime": _now_iso(),
            "model": {"detector": "YOLOv8", "tracker": "BoT-SORT", "reid": "OSNet"},
        }

    @app.get("/api/dashboard")
    def dashboard():
        seats = _all_seats()
        _detect_events(seats)
        return {
            "serverTime": _now_iso(),
            "summary":    _build_summary(seats, events.count_unconfirmed()),
            "settings":   settings,
            "seats":      seats,
            "events":     events.get_events(limit=20),
        }

    @app.get("/api/seats")
    def get_seats(includeEmpty: bool = True):
        seats = _all_seats()
        if not includeEmpty:
            seats = [s for s in seats if s["occupancyState"] != "EMPTY"]
        return {"seats": seats}

    @app.get("/api/seats/layout")
    def get_layout():
        return {
            "cameraId":    "main",
            "imageWidth":  img_w,
            "imageHeight": img_h,
            "seats": [
                {"seatId": cfg["seatId"], "label": cfg["label"], "roi": cfg["roi"]}
                for cfg in seat_configs.values()
            ],
        }

    @app.get("/api/seats/{seat_id}")
    def get_seat(seat_id: str):
        cfg = seat_configs.get(seat_id)
        if cfg is None:
            raise HTTPException(
                status_code=404,
                detail={"error": {"code": "INVALID_SEAT_ID",
                                  "message": "존재하지 않는 좌석입니다.",
                                  "details": {"seatId": seat_id}}},
            )
        gallery_map = {e["seatId"]: e for e in gallery.get_status()}
        return {"seat": _build_seat(cfg, gallery_map.get(seat_id))}

    @app.get("/api/events")
    def get_events(
        status:  Optional[str] = None,
        seatId:  Optional[str] = None,
        limit:   int = 20,
    ):
        return {"events": events.get_events(status=status, seat_id=seatId, limit=limit)}

    @app.post("/api/events/{event_id}/action")
    def event_action(event_id: str, body: dict):
        action = body.get("action")
        memo   = body.get("memo")
        result = events.update_status(event_id, action, memo)
        if result is None:
            raise HTTPException(status_code=404,
                                detail={"error": {"code": "EVENT_NOT_FOUND",
                                                  "message": "이벤트를 찾을 수 없습니다."}})
        return {"event": result}

    @app.get("/api/gallery")
    def get_gallery():
        return {"persons": snapshot_store.get_all()}

    @app.get("/api/seats/{seat_id}/snapshot")
    def get_seat_snapshot(seat_id: str):
        snaps = snapshot_store.get_by_seat(seat_id)
        if not snaps:
            raise HTTPException(status_code=404, detail="스냅샷 없음")
        return {"snapshots": snaps}

    @app.get("/api/settings")
    def get_settings():
        return {"settings": settings}

    @app.patch("/api/settings")
    def patch_settings(body: dict):
        allowed = set(_DEFAULT_SETTINGS.keys())
        for k, v in body.items():
            if k in allowed:
                settings[k] = v
                # gallery 런타임 임계값 동기화
                if k == "useLimitSeconds":
                    gallery.alert_threshold = float(v)
                elif k == "nearLimitBeforeSeconds":
                    gallery.near_limit_threshold = float(v)
                elif k == "awayThresholdSeconds":
                    gallery.away_threshold = float(v)
        return {"settings": settings}

    # ── MJPEG 스트림 ─────────────────────────────────────────────────────

    def _mjpeg_generator(overlay: bool):
        while True:
            frame = _frame_buffer.get()
            if frame is None:
                time.sleep(0.05)
                continue
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )
            time.sleep(0.033)  # ~30fps 상한

    @app.get("/api/cameras/main/stream")
    def stream(overlay: bool = False):
        return StreamingResponse(
            _mjpeg_generator(overlay),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # ── WebSocket ─────────────────────────────────────────────────────────

    async def _broadcast(msg: dict) -> None:
        dead = []
        for ws in ws_clients:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            ws_clients.remove(ws)

    async def _ws_broadcaster() -> None:
        """상태 변경 시에만 seat.updated 브로드캐스트.

        accumulatedSeconds는 프론트가 자체 타이머로 증가시킨다 (AI_HANDOFF_CONTEXT 11.2).
        서버는 occupancyState / alertState / belongings 변화 시에만 발송한다.
        """
        # seatId → (occupancyState, alertState, belongings_key)
        prev_sig: dict[str, tuple] = {}
        heartbeat_tick = 0

        while True:
            await asyncio.sleep(1)
            heartbeat_tick += 1

            seats = _all_seats()
            _detect_events(seats)
            now = _now_iso()

            if ws_clients:
                for s in seats:
                    sid = s["seatId"]
                    sig = (
                        s["occupancyState"],
                        s["alertState"],
                        str(s.get("belongings")),   # 짐 목록 변화도 감지
                    )
                    if prev_sig.get(sid) != sig:
                        prev_sig[sid] = sig
                        await _broadcast({"type": "seat.updated", "serverTime": now, "seat": s})

                # 신규 이벤트 발송
                for evt in events.pop_pending():
                    await _broadcast({"type": "event.created", "serverTime": now, "event": evt})

                # heartbeat 10초마다
                if heartbeat_tick % 10 == 0:
                    await _broadcast({"type": "heartbeat", "serverTime": now})

    @app.websocket("/ws/seats")
    async def ws_seats(websocket: WebSocket):
        await websocket.accept()
        ws_clients.append(websocket)
        try:
            # 연결 직후 snapshot 전송
            seats = _all_seats()
            await websocket.send_json({
                "type":       "snapshot",
                "serverTime": _now_iso(),
                "summary":    _build_summary(seats, events.count_unconfirmed()),
                "seats":      seats,
                "events":     events.get_events(limit=20),
            })
            while True:
                await websocket.receive_text()   # 클라이언트 메시지 대기 (연결 유지)
        except WebSocketDisconnect:
            pass
        finally:
            if websocket in ws_clients:
                ws_clients.remove(websocket)

    @app.on_event("startup")
    async def _startup():
        asyncio.create_task(_ws_broadcaster())

    return app


# ── 기본값 ────────────────────────────────────────────────────────────────────

_DEFAULT_SETTINGS: dict = {
    "useLimitSeconds":        7200,
    "nearLimitBeforeSeconds": 600,
    "awayThresholdSeconds":   300,
    "leftGraceSeconds":       60,
    "minSeatIou":             0.35,
    "eventDebounceSeconds":   10,
}
