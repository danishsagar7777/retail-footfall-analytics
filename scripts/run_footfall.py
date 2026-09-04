import os
import time
import yaml
from dataclasses import dataclass

from ultralytics import YOLO

from src.input.rtsp_reader import RTSPReader
from src.input.frame_processor import resize_frame
from src.analytics.pipeline import FootfallPipeline
from src.utils.fps import FPSCounter


@dataclass
class SimpleTrack:
    track_id: int
    bbox: tuple
    confidence: float

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox

        return (
            (x1 + x2) / 2,
            (y1 + y2) / 2,
        )


def load_config(path="configs/config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def main():

    config = load_config()

    rtsp_url = os.getenv("RTSP_URL")

    if not rtsp_url:
        raise RuntimeError(
            "RTSP_URL environment variable is not set.\n"
            "Run: export RTSP_URL='your_rtsp_url'"
        )

    detection_config = config["detection"]
    analytics_config = config["analytics"]
    tracking_config = config["tracking"]
    output_config = config["output"]

    model_path = detection_config.get(
        "model",
        "yolov8n_openvino_model",
    )

    confidence = detection_config.get(
        "confidence",
        0.40,
    )

    classes = detection_config.get(
        "classes",
        [0],
    )

    imgsz = detection_config.get(
        "imgsz",
        640,
    )

    counting_line = analytics_config[
        "counting_line"
    ]

    line_buffer = analytics_config.get(
        "line_buffer",
        12.0,
    )

    event_cooldown = analytics_config.get(
        "event_cooldown",
        2.0,
    )

    log_file = output_config.get(
        "log_file",
        "logs/footfall_events.csv",
    )

    log_directory = os.path.dirname(log_file)

    if log_directory:
        os.makedirs(
            log_directory,
            exist_ok=True,
        )

    print("=" * 60)
    print("RETAIL FOOTFALL ANALYTICS")
    print("=" * 60)

    print(
        f"Loading model: {model_path}"
    )

    model = YOLO(model_path)

    print(
        f"Confidence threshold: {confidence}"
    )

    print(
        f"Image size: {imgsz}"
    )

    print(
        f"Classes: {classes}"
    )

    print(
        f"Counting line: {counting_line}"
    )

    print(
        f"Line buffer: {line_buffer}px"
    )

    print(
        f"Event cooldown: {event_cooldown}s"
    )

    print(
        f"Event log: {log_file}"
    )

    reader = RTSPReader(rtsp_url)

    pipeline = FootfallPipeline(
        counting_line=counting_line,
        log_file=log_file,
        event_cooldown=event_cooldown,
        line_buffer=line_buffer,
    )

    fps_counter = FPSCounter()

    reader.start()

    frame_count = 0

    print("=" * 60)
    print("LIVE FOOTFALL PIPELINE STARTED")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    try:

        while True:

            ret, frame = reader.read()

            if not ret:
                time.sleep(0.001)
                continue

            # Prepare frame for YOLO.
            processed_frame = resize_frame(
                frame,
                width=imgsz,
                height=imgsz,
            )

            results = model.track(
                processed_frame,
                persist=True,
                tracker=tracking_config.get(
                    "tracker",
                    "botsort.yaml",
                ),
                classes=classes,
                conf=confidence,
                imgsz=imgsz,
                verbose=False,
            )

            result = results[0]

            tracks = []

            if result.boxes is not None:

                boxes = result.boxes

                if boxes.id is not None:

                    ids = (
                        boxes.id
                        .cpu()
                        .tolist()
                    )

                    xyxy = (
                        boxes.xyxy
                        .cpu()
                        .tolist()
                    )

                    confs = (
                        boxes.conf
                        .cpu()
                        .tolist()
                    )

                    for (
                        track_id,
                        bbox,
                        track_confidence,
                    ) in zip(
                        ids,
                        xyxy,
                        confs,
                    ):

                        tracks.append(
                            SimpleTrack(
                                track_id=int(
                                    track_id
                                ),
                                bbox=tuple(
                                    bbox
                                ),
                                confidence=float(
                                    track_confidence
                                ),
                            )
                        )

            events = pipeline.update(
                tracks
            )

            fps_counter.update()

            frame_count += 1

            for event in events:

                print(
                    "EVENT: "
                    f"track_id={event['track_id']} "
                    f"type={event['event']}"
                )

            if frame_count % 100 == 0:

                summary = (
                    pipeline.summary()
                )

                print(
                    f"Frames: {frame_count} | "
                    f"FPS: {fps_counter.fps:.2f} | "
                    f"Entries: {summary['entries']} | "
                    f"Exits: {summary['exits']} | "
                    f"Occupancy: {summary['occupancy']} | "
                    f"Reconnects: "
                    f"{reader.reconnect_count} | "
                    f"Skipped: "
                    f"{reader.dropped_frames}"
                )

    except KeyboardInterrupt:

        print(
            "\nStopping pipeline..."
        )

    finally:

        reader.release()

        summary = pipeline.summary()

        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)

        print(
            f"Entries: "
            f"{summary['entries']}"
        )

        print(
            f"Exits: "
            f"{summary['exits']}"
        )

        print(
            f"Occupancy: "
            f"{summary['occupancy']}"
        )

        print(
            f"Average FPS: "
            f"{fps_counter.fps:.2f}"
        )

        print(
            f"RTSP reconnects: "
            f"{reader.reconnect_count}"
        )

        print(
            f"Skipped frames: "
            f"{reader.dropped_frames}"
        )

        print(
            f"Events logged to: "
            f"{log_file}"
        )

        print("=" * 60)


if __name__ == "__main__":
    main()
