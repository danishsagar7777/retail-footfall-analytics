import os
import yaml

from src.input.rtsp_reader import RTSPReader
from src.input.frame_processor import resize_frame
from src.tracking.tracker import PersonTracker
from src.analytics.pipeline import FootfallPipeline
from src.utils.fps import FPSCounter


def load_config():
    with open("configs/config.yaml", "r") as file:
        return yaml.safe_load(file)


def main():

    # --------------------------------------------------
    # Load configuration
    # --------------------------------------------------

    config = load_config()

    counting_line = tuple(
        config["analytics"]["counting_line"]
    )

    # --------------------------------------------------
    # RTSP URL
    # --------------------------------------------------

    rtsp_url = os.getenv("RTSP_URL")

    if not rtsp_url:
        raise RuntimeError("RTSP_URL is not set.")

    # --------------------------------------------------
    # Initialize components
    # --------------------------------------------------

    reader = RTSPReader(rtsp_url)

    tracker = PersonTracker()

    pipeline = FootfallPipeline(
        counting_line=counting_line
    )

    fps_counter = FPSCounter()

    # --------------------------------------------------
    # Connect
    # --------------------------------------------------

    print("Connecting to RTSP stream...")

    reader.connect()

    print("RTSP stream connected.")
    print("Starting footfall analytics...")
    print("Press Ctrl+C to stop.")

    frame_count = 0

    try:

        while True:

            # --------------------------------------------------
            # Read frame
            # --------------------------------------------------

            ret, frame = reader.read()

            if not ret:
                print("Failed to receive frame.")
                continue

            # --------------------------------------------------
            # Resize
            # --------------------------------------------------

            frame = resize_frame(frame)

            # --------------------------------------------------
            # Tracking
            # --------------------------------------------------

            tracks = tracker.update(frame)

            # --------------------------------------------------
            # Footfall analytics
            # --------------------------------------------------

            events = pipeline.update(tracks)

            # --------------------------------------------------
            # FPS
            # --------------------------------------------------

            fps_counter.update()

            frame_count += 1

            # --------------------------------------------------
            # Events
            # --------------------------------------------------

            for event in events:

                print(
                    f"EVENT | "
                    f"Track ID={event['track_id']} | "
                    f"{event['event'].upper()}"
                )

            # --------------------------------------------------
            # Statistics
            # --------------------------------------------------

            if frame_count % 100 == 0:

                summary = pipeline.summary()

                print()
                print(
                    f"FRAME      : {frame_count}"
                )

                print(
                    f"FPS        : {fps_counter.fps:.2f}"
                )

                print(
                    f"ENTRIES    : {summary['entries']}"
                )

                print(
                    f"EXITS      : {summary['exits']}"
                )

                print(
                    f"OCCUPANCY  : {summary['occupancy']}"
                )

                print()

    except KeyboardInterrupt:

        print("\nStopping...")

        summary = pipeline.summary()

        print()
        print("Final Footfall Summary")
        print("----------------------")
        print(f"Frames     : {frame_count}")
        print(f"FPS        : {fps_counter.fps:.2f}")
        print(f"Entries    : {summary['entries']}")
        print(f"Exits      : {summary['exits']}")
        print(f"Occupancy  : {summary['occupancy']}")

    finally:

        reader.release()


if __name__ == "__main__":
    main()
