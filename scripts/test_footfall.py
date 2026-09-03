import os
import time
import yaml

from src.input.rtsp_reader import RTSPReader
from src.input.frame_processor import resize_frame
from src.tracking.tracker import PersonTracker
from src.analytics.pipeline import FootfallPipeline
from src.utils.fps import FPSCounter


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

    reader = RTSPReader(rtsp_url)

    tracker = PersonTracker(
        model_path=config["detection"]["model"],
        tracker_config=config["tracking"]["tracker"],
        confidence=confidence,
    )

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
        "Starting PyTorch footfall pipeline..."
    )

    print(f"Log file: {log_file}")
    print(f"Event cooldown: {event_cooldown}s")
    print(f"Line buffer: {line_buffer}px")

    try:
        while True:
            ret, frame = reader.read()

            if not ret:
                time.sleep(0.001)
                continue

            frame = resize_frame(frame)

            tracks = tracker.update(frame)

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
                    f"{reader.reconnect_count}"
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
            f"Events logged to: "
            f"{log_file}"
        )


if __name__ == "__main__":
    main()
