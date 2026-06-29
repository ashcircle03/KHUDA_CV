from __future__ import annotations
import cv2
import numpy as np


class Camera:
    def __init__(
        self,
        source: int | str,
        width:  int = 1280,
        height: int = 720,
        fps:    int = 10,
    ) -> None:
        self._cap = cv2.VideoCapture(source)
        if isinstance(source, str):
            # 영상 파일은 원본 FPS 그대로 사용
            pass
        else:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS,          fps)

        if not self._cap.isOpened():
            raise RuntimeError(f"카메라/영상 열기 실패: {source}")

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read(self) -> np.ndarray | None:
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self) -> None:
        self._cap.release()
