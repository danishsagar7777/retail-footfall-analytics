from ultralytics import YOLO

from src.tracking.track import Track


class PersonTracker:
    def __init__(
        self,
        model_path="yolov8n.pt",
        tracker_config="botsort.yaml",
        confidence=0.40,
    ):
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.confidence = confidence

    def update(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            classes=[0],
            conf=self.confidence,
            verbose=False,
        )

        result = results[0]

        tracks = []

        # No bounding boxes
        if result.boxes is None:
            return tracks

        # Boxes exist, but tracker has not assigned IDs
        if result.boxes.id is None:
            return tracks

        boxes = result.boxes.xyxy.cpu().tolist()
        ids = result.boxes.id.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()

        for bbox, track_id, confidence in zip(
            boxes,
            ids,
            confidences,
        ):
            tracks.append(
                Track(
                    track_id=track_id,
                    bbox=tuple(bbox),
                    confidence=confidence,
                )
            )

        return tracks
