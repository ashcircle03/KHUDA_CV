"""좌석 ROI 설정 도구 — 창 안에서 모든 조작 완결.

조작:
  마우스 드래그  좌석 영역 그리기
  숫자/영문 입력 좌석 ID 타이핑 (창 안에서)
  Enter          현재 ROI 확정
  Backspace      ID 한 글자 지우기
  r              마지막 ROI 삭제
  s              rois.json 저장 후 종료
  q              저장 없이 종료
"""
import json
import cv2
import numpy as np

VIDEO   = "cafe_cctv.mp4"
OUTPUT  = "rois.json"
WIN_MAX = 1280   # 화면에 표시할 최대 너비

# ── 상태 ──────────────────────────────────────────────────────────────────────
rois:      dict[str, list] = {}
drawing    = False
start_pt   = (0, 0)
cur_rect   = None
pending    = None
typed_id   = ""
frame_orig = None
scale      = 1.0   # 윈도우/원본 비율 (마우스 좌표 → 원본 좌표 변환용)

COLORS = [(0,200,100),(100,150,255),(0,180,255),(255,160,0),(200,80,255)]


def load_existing() -> dict:
    try:
        with open(OUTPUT) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def to_img(x: int, y: int):
    """윈도우 좌표 → 원본 이미지 좌표 변환."""
    return int(x / scale), int(y / scale)


def to_win(x: int, y: int):
    """원본 이미지 좌표 → 윈도우 좌표 변환 (오버레이 표시용)."""
    return int(x * scale), int(y * scale)


def draw_frame(disp: np.ndarray) -> np.ndarray:
    """disp는 이미 윈도우 크기로 리사이즈된 프레임."""
    vis = disp.copy()
    h, w = vis.shape[:2]

    for i, (sid, pts) in enumerate(rois.items()):
        col   = COLORS[i % len(COLORS)]
        # 저장된 좌표는 원본 기준 → 윈도우 기준으로 변환해서 그림
        w_pts = [to_win(p[0], p[1]) for p in pts]
        poly  = np.array(w_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [poly], True, col, 2)
        cv2.putText(vis, sid, (w_pts[0][0]+4, w_pts[0][1]+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    if cur_rect:
        cv2.rectangle(vis, cur_rect[0], cur_rect[1], (255,255,0), 2)

    if pending:
        pt1, pt2 = pending
        cv2.rectangle(vis, pt1, pt2, (0,255,255), 2)
        label = f"ID: {typed_id}_"
        cv2.putText(vis, label, (pt1[0], pt1[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)

    hint = (
        f"좌석 ID 입력 후 Enter  |  현재: \"{typed_id}_\"  |  Backspace=지우기  Esc=취소"
        if pending else
        f"ROI {len(rois)}개  |  드래그=그리기  r=마지막삭제  s=저장  q=종료"
    )
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, h-32), (w, h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
    cv2.putText(vis, hint, (10, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220,220,220), 1)
    return vis


def mouse_cb(event, x, y, flags, _):
    global drawing, start_pt, cur_rect, pending, typed_id

    if pending:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing  = True
        start_pt = (x, y)
        cur_rect = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cur_rect = (start_pt, (x, y))

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing  = False
        x1, y1   = min(start_pt[0], x), min(start_pt[1], y)
        x2, y2   = max(start_pt[0], x), max(start_pt[1], y)
        cur_rect = None
        if abs(x2-x1) < 5 or abs(y2-y1) < 5:
            return
        pending  = ((x1, y1), (x2, y2))
        typed_id = ""


def main():
    global frame_orig, rois, scale, pending, typed_id

    rois = load_existing()
    if rois:
        print(f"기존 ROI 로드: {list(rois.keys())}")

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"영상 열기 실패: {VIDEO}"); return

    ret, frame_orig = cap.read()
    cap.release()
    if not ret:
        print("프레임 읽기 실패"); return

    img_h, img_w = frame_orig.shape[:2]

    # 윈도우 크기 계산 (원본 비율 유지)
    scale   = min(1.0, WIN_MAX / img_w)
    win_w   = int(img_w * scale)
    win_h   = int(img_h * scale)
    disp    = cv2.resize(frame_orig, (win_w, win_h))

    print(f"원본 해상도: {img_w}×{img_h}  |  표시 해상도: {win_w}×{win_h}  |  scale={scale:.3f}")
    print("마우스 좌표는 자동으로 원본 해상도로 변환됩니다.\n")

    cv2.namedWindow("ROI 설정", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("ROI 설정", mouse_cb)

    while True:
        cv2.imshow("ROI 설정", draw_frame(disp))
        key = cv2.waitKey(30) & 0xFF

        if key == 0xFF:
            continue

        # ── ID 입력 모드 ──────────────────────────────────────────────
        if pending:
            if key == 13:        # Enter
                if typed_id:
                    pt1, pt2 = pending
                    # 윈도우 좌표 → 원본 좌표로 변환해서 저장
                    ix1, iy1 = to_img(*pt1)
                    ix2, iy2 = to_img(*pt2)
                    rois[typed_id] = [[ix1,iy1],[ix2,iy1],[ix2,iy2],[ix1,iy2]]
                    print(f"  ✓ {typed_id}  원본좌표: ({ix1},{iy1})-({ix2},{iy2})")
                pending  = None
                typed_id = ""
            elif key == 8:
                typed_id = typed_id[:-1]
            elif key == 27:
                pending  = None
                typed_id = ""
            elif 32 <= key < 127:
                typed_id += chr(key)
            continue

        # ── 일반 모드 ─────────────────────────────────────────────────
        if key == ord('s'):
            with open(OUTPUT, "w") as f:
                json.dump(rois, f, indent=2, ensure_ascii=False)
            print(f"\n저장 완료 → {OUTPUT}")
            print(json.dumps(rois, indent=2, ensure_ascii=False))
            break
        elif key == ord('r') and rois:
            removed = list(rois.keys())[-1]
            del rois[removed]
            print(f"  삭제: {removed}")
        elif key == ord('q'):
            print("저장 없이 종료")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
