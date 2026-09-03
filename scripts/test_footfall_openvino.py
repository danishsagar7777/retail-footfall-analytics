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

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        return cx, cy


def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():

    config = load_config()

    rtsp_url = os.getenv("RTSP_URL")

    if not rtsp_url:
        raise RuntimeError(
            "RTSP_URL environment variable is not set"
        )

    confidence = config["detection"]["confidence"]
    classes = config["detection"]["classes"]
    imgsz = config["detection"].get("imgsz", 640)

    counting_line = config["analytics"]["counting_line"]

    line_buffer = config["analytics"].get(
        "line_buffer",
        12.0,
    )

    event_cooldown = config["analytics"].get(
        "event_cooldown",
        2.0,
    )

    log_file = config["output"]["log_file"]

    model_path = "yolov8n_openvino_model"

    print(
        f"Loading OpenVINO model: {model_path}"
    )

    model = YOLO(model_path)

    reader = RTSPReader(rtsp_url)

    pipeline = FootfallPipeline(
        counting_line,
        log_file=log_file,
        event_cooldown=event_cooldown,
        line_buffer=line_buffer,
    )

    fps_counter = FPSCounter()

    reader.start()

    frame_count = 0

    print(
        "Starting optimized OpenVINO "
        "footfall pipeline..."
    )

    print(f"Log file: {log_file}")
    print(f"YOLO image size: {imgsz}")
    print(f"Confidence: {confidence}")
    print(f"Event cooldown: {event_cooldown}s")
    print(f"Line buffer: {line_buffer}px")

    try:

        while True:

            ret, frame = reader.read()

            if not ret:
                time.sleep(0.001)
                continue

            frame = resize_frame(frame)

            results = model.track(
                frame,
                persist=True,
                tracker=config["tracking"]["tracker"],
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

                    ids = boxes.id.cpu().tolist()
                    xyxy = boxes.xyxy.cpu().tolist()
                    confs = boxes.conf.cpu().tolist()

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
                                track_id=int(track_id),
                                bbox=tuple(bbox),
                                confidence=float(
                                    track_confidence
                                ),
                            )
                        )

            events = pipeline.update(tracks)

            fps_counter.update()
            frame_count += 1

            for event in events:

                print(
                    f"EVENT: "
                    f"track_id={event['track_id']} "
                    f"type={event['event']}"
                )

            if frame_count % 100 == 0:

                summary = pipeline.summary()

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

        print("\nStopping pipeline...")

    finally:

        reader.release()

        summary = pipeline.summary()

        print("\nFinal summary:")
        print(summary)

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


if __name__ == "__main__":
    main()
