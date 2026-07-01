"""YOLO person detector wrapper.

테이블/짐 점유 여부는 table_change.py의 baseline 기반 변화 감지가 담당한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Box:
    xyxy: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    cls_name: str = ""
    keypoints: np.ndarray | None = None  # TEMP DEBUG: [17, 2] COCO pose keypoints, 테스트 후 제거 예정


@dataclass
class DetectionResult:
    person_boxes: list[Box]


class Detector:
    def __init__(
        self,
        model: str = "yolov8s-pose.pt",  # TEMP DEBUG: pose 모델 테스트용, 원래는 yolov8s.pt
        person_conf: float = 0.25,
        imgsz: int = 448,
        **_: object,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(model)
        self._person_conf = person_conf
        self._imgsz = imgsz

    def detect_person_only(self, frame: np.ndarray) -> DetectionResult:
        """사람만 감지한다. COCO class 0(person)만 추론한다."""
        results = self._model.predict(
            frame,
            conf=self._person_conf,
            classes=[0],
            imgsz=self._imgsz,
            verbose=False,
        )
        persons: list[Box] = []
        for r in results:
            # TEMP DEBUG: pose 모델일 때만 keypoints 존재, 테스트 후 제거 예정
            keypoints_xy = (
                r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
            )
            for i, box in enumerate(r.boxes):
                confidence = float(box.conf[0])
                if confidence < self._person_conf:
                    continue
                kpts = keypoints_xy[i] if keypoints_xy is not None else None
                persons.append(Box(
                    xyxy=box.xyxy[0].cpu().numpy(),
                    confidence=confidence,
                    cls_name="person",
                    keypoints=kpts,
                ))
        return DetectionResult(person_boxes=persons)
