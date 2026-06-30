"""감지 래퍼.

- 사람: YOLO COCO detector(person class only)
- 테이블/짐 점유 여부: table_change.py의 baseline 기반 변화 감지가 담당
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Box:
    xyxy: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    cls_name: str = ""


@dataclass
class DetectionResult:
    person_boxes: list[Box]
    luggage_boxes: list[Box]


# tracker.py의 legacy belongings 변환 호환용. 새 상태엔진은 이 값을 쓰지 않는다.
_BELONGING_MAP: dict[str, dict] = {
    "cup": {"type": "CUP", "label": "컵"},
    "coffee cup": {"type": "CUP", "label": "커피컵"},
    "paper cup": {"type": "CUP", "label": "종이컵"},
    "mug": {"type": "CUP", "label": "머그컵"},
    "tumbler": {"type": "CUP", "label": "텀블러"},
    "bottle": {"type": "CUP", "label": "음료"},
    "water bottle": {"type": "CUP", "label": "물병"},
    "plastic bottle": {"type": "CUP", "label": "페트병"},
    "drink bottle": {"type": "CUP", "label": "음료병"},
    "laptop": {"type": "LAPTOP", "label": "노트북"},
    "laptop computer": {"type": "LAPTOP", "label": "노트북"},
    "backpack": {"type": "BACKPACK", "label": "백팩"},
    "bag": {"type": "BACKPACK", "label": "가방"},
    "handbag": {"type": "HANDBAG", "label": "핸드백"},
}


def belonging_meta(cls_name: str) -> dict:
    return _BELONGING_MAP.get(cls_name, {"type": "UNKNOWN", "label": cls_name})


class Detector:
    def __init__(
        self,
        model: str = "yolov8s.pt",
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
            for box in r.boxes:
                confidence = float(box.conf[0])
                if confidence < self._person_conf:
                    continue
                persons.append(Box(
                    xyxy=box.xyxy[0].cpu().numpy(),
                    confidence=confidence,
                    cls_name="person",
                ))
        return DetectionResult(person_boxes=persons, luggage_boxes=[])

    def detect(self, frame: np.ndarray, augment_rois: bool = False) -> DetectionResult:
        """Legacy 호출 호환용. 새 로직에서는 짐 탐지를 수행하지 않는다."""
        return self.detect_person_only(frame)
