"""테이블/좌석 ROI 폴리곤 설정 도구.

조작:
  좌클릭          현재 polygon 꼭짓점 추가
  Enter           테이블 polygon 확정 -> 좌석 polygon 입력 -> ID 입력/저장
  Backspace       마지막 꼭짓점 또는 ID 한 글자 삭제
  r               마지막 저장 ROI 쌍 삭제
  Esc             현재 테이블/좌석 입력 취소
  s               rois.json 저장 후 종료
  q               저장 없이 종료
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from roi_utils import RoiConfig, SeatPolygon

VIDEO = "cafe_cctv.mp4"
OUTPUT = "rois.json"
WIN_MAX = 1280

rois: list[SeatPolygon] = []
points: list[tuple[int, int]] = []
pending_table: list[dict[str, float]] | None = None
typed_id = ""
id_mode = False
phase = "table"
frame_orig = None
scale = 1.0

SEAT_COLORS = [(0, 200, 100), (100, 150, 255), (0, 180, 255), (255, 160, 0), (200, 80, 255)]
TABLE_COLOR = (0, 210, 255)
ACTIVE_TABLE_COLOR = (0, 240, 255)
ACTIVE_SEAT_COLOR = (80, 255, 130)


def to_img(x: int, y: int) -> tuple[int, int]:
    return int(x / scale), int(y / scale)


def to_win(x: float, y: float) -> tuple[int, int]:
    return int(x * scale), int(y * scale)


def load_existing(width: int, height: int) -> list[SeatPolygon]:
    config = RoiConfig.load(OUTPUT, width, height)
    return list(config.seats)


def _polygon_to_window(polygon: list[dict[str, float]]) -> np.ndarray:
    return np.array(
        [to_win(p["x"] * frame_orig.shape[1], p["y"] * frame_orig.shape[0]) for p in polygon],
        dtype=np.int32,
    ).reshape(-1, 1, 2)


def draw_frame(disp: np.ndarray) -> np.ndarray:
    vis = disp.copy()
    h, _ = vis.shape[:2]

    for i, seat in enumerate(rois):
        seat_col = SEAT_COLORS[i % len(SEAT_COLORS)]
        table_poly = _polygon_to_window(seat.table_polygon)
        seat_poly = _polygon_to_window(seat.seat_polygon)
        cv2.polylines(vis, [table_poly], True, TABLE_COLOR, 1)
        cv2.polylines(vis, [seat_poly], True, seat_col, 2)
        label_pt = tuple(seat_poly.reshape(-1, 2)[0])
        cv2.putText(vis, seat.seat_id, (label_pt[0] + 4, label_pt[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, seat_col, 2)

    if pending_table:
        table_poly = _polygon_to_window(pending_table)
        cv2.polylines(vis, [table_poly], True, ACTIVE_TABLE_COLOR, 2)
        cv2.putText(vis, "table ok", tuple(table_poly.reshape(-1, 2)[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ACTIVE_TABLE_COLOR, 2)

    if points:
        active_color = ACTIVE_TABLE_COLOR if phase == "table" else ACTIVE_SEAT_COLOR
        poly = np.array(points, dtype=np.int32)
        for p in points:
            cv2.circle(vis, p, 4, active_color, -1)
        if len(points) > 1:
            cv2.polylines(vis, [poly.reshape(-1, 1, 2)], False, active_color, 2)
        if len(points) >= 3:
            cv2.polylines(vis, [poly.reshape(-1, 1, 2)], True, active_color, 1)

    hint = _hint_text()
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, h - 34), (vis.shape[1], h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
    cv2.putText(vis, hint, (10, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1)
    return vis


def _hint_text() -> str:
    if id_mode:
        return f"ID 입력 후 Enter  |  현재: \"{typed_id}_\"  |  Backspace=지우기  Esc=취소"
    target = "테이블" if phase == "table" else "좌석"
    return (
        f"ROI 쌍 {len(rois)}개  |  현재={target} polygon  |  "
        "클릭=꼭짓점  Enter=확정  Backspace=점삭제  r=마지막삭제  s=저장  q=종료"
    )


def mouse_cb(event, x, y, flags, _):
    if id_mode:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))


def save_rois(width: int, height: int) -> None:
    config = RoiConfig(rois, width, height)
    with open(OUTPUT, "w") as f:
        json.dump(config.to_json(), f, indent=2, ensure_ascii=False)
    print(f"\n저장 완료 -> {OUTPUT}")
    print(json.dumps(config.to_json(), indent=2, ensure_ascii=False))


def _normalize_points(width: int, height: int) -> list[dict[str, float]]:
    normalized = []
    for x, y in points:
        ix, iy = to_img(x, y)
        normalized.append({
            "x": round(max(0.0, min(1.0, ix / max(width, 1))), 6),
            "y": round(max(0.0, min(1.0, iy / max(height, 1))), 6),
        })
    return normalized


def confirm_current_polygon(width: int, height: int) -> None:
    global points, pending_table, phase, id_mode, typed_id
    if len(points) < 3:
        return
    if phase == "table":
        pending_table = _normalize_points(width, height)
        points = []
        phase = "seat"
        print(f"  table polygon 확정: {len(pending_table)} points")
        return
    id_mode = True
    typed_id = ""


def confirm_pair(width: int, height: int) -> None:
    global points, pending_table, typed_id, id_mode, phase
    if len(points) < 3 or not typed_id or not pending_table:
        return
    seat_polygon = _normalize_points(width, height)
    rois.append(SeatPolygon(typed_id, typed_id, pending_table, seat_polygon))
    print(f"  + {typed_id}: table {len(pending_table)} points / seat {len(seat_polygon)} points")
    points = []
    pending_table = None
    typed_id = ""
    id_mode = False
    phase = "table"


def cancel_current_pair() -> None:
    global points, pending_table, typed_id, id_mode, phase
    points = []
    pending_table = None
    typed_id = ""
    id_mode = False
    phase = "table"


def main() -> None:
    global frame_orig, rois, scale, typed_id, id_mode, points

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"영상 열기 실패: {VIDEO}")
        return
    ret, frame_orig = cap.read()
    cap.release()
    if not ret:
        print("프레임 읽기 실패")
        return

    img_h, img_w = frame_orig.shape[:2]
    rois = load_existing(img_w, img_h)
    if rois:
        print(f"기존 ROI 로드: {[s.seat_id for s in rois]}")

    scale = min(1.0, WIN_MAX / img_w)
    win_w, win_h = int(img_w * scale), int(img_h * scale)
    disp = cv2.resize(frame_orig, (win_w, win_h))
    print(f"원본 해상도: {img_w}x{img_h} | 표시 해상도: {win_w}x{win_h} | scale={scale:.3f}")
    print("각 좌석마다 테이블 polygon을 먼저 찍고 Enter, 이어서 좌석 polygon을 찍고 Enter 후 ID를 입력합니다.")

    cv2.namedWindow("테이블-좌석 ROI 폴리곤 설정", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("테이블-좌석 ROI 폴리곤 설정", mouse_cb)

    while True:
        cv2.imshow("테이블-좌석 ROI 폴리곤 설정", draw_frame(disp))
        key = cv2.waitKey(30) & 0xFF
        if key == 0xFF:
            continue

        if id_mode:
            if key == 13:
                confirm_pair(img_w, img_h)
            elif key == 8:
                typed_id = typed_id[:-1]
            elif key == 27:
                id_mode = False
                typed_id = ""
            elif 32 <= key < 127:
                typed_id += chr(key)
            continue

        if key == 13:
            confirm_current_polygon(img_w, img_h)
        elif key == 8 and points:
            points.pop()
        elif key == 27:
            cancel_current_pair()
        elif key == ord("r") and rois:
            removed = rois.pop()
            print(f"  삭제: {removed.seat_id}")
        elif key == ord("s"):
            save_rois(img_w, img_h)
            break
        elif key == ord("q"):
            print("저장 없이 종료")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
