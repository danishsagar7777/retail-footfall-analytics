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

        if result.boxes is None:
            return tracks

        boxes = result.boxes

        if boxes.id is None:
            return tracks

        ids = boxes.id.cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()

        for track_id, bbox, confidence in zip(
            ids,
            xyxy,
            confs,
        ):

            tracks.append(
                Track(
                    track_id=int(track_id),
                    bbox=tuple(bbox),
                    confidence=float(confidence),
                )
            )

        return tracks
