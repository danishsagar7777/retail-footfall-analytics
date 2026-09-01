from dataclasses import dataclass
from typing import Tuple


@dataclass
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        return cx, cy
