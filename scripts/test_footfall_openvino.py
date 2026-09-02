import os
import yaml

from ultralytics import YOLO

from src.input.rtsp_reader import RTSPReader
from src.input.frame_processor import resize_frame
from src.analytics.pipeline import FootfallPipeline
from src.utils.fps import FPSCounter


def load_config():
    with open("configs/config.yaml", "r") as file:
        return yaml.safe_load(file)


def main():

    # Load configuration
    config = load_config()

    counting_line = tuple(
        config["analytics"]["counting_line"]
    )

    confidence = config["detection"]["confidence"]

    # RTSP URL
    rtsp_url = os.getenv("RTSP_URL")

    if not rtsp_url:
        raise RuntimeError("RTSP_URL is not set.")

    # Load OpenVINO model
    print("Loading OpenVINO YOLO model...")

    model = YOLO("yolov8n_openvino_model")

    # Initialize components
    reader = RTSPReader(rtsp_url)

    pipeline = FootfallPipeline(
        counting_line=counting_line
    )

    fps_counter = FPSCounter()

    # Connect to RTSP
    print("Connecting to RTSP stream...")

    reader.connect()

    print("RTSP stream connected.")
    print("Starting OpenVINO footfall pipeline...")
    print("Press Ctrl+C to stop.")

    frame_count = 0

    try:

        while True:

            # Read frame
            ret, frame = reader.read()

            if not ret:
                print("Failed to receive frame.")
                continue

            # Resize to 1280x720
            frame = resize_frame(frame)

            # OpenVINO YOLO + BoT-SORT
            results = model.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                classes=[0],
                conf=confidence,
                verbose=False,
            )

            result = results[0]

            tracks = []

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):

                ids = result.boxes.id.cpu().tolist()
                boxes = result.boxes.xyxy.cpu().tolist()

                for track_id, bbox in zip(ids, boxes):

                    x1, y1, x2, y2 = bbox

                    center = (
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                    )

                    class SimpleTrack:

                        def __init__(
                            self,
                            track_id,
                            center,
                        ):
                            self.track_id = int(track_id)
                            self.center = center

                    tracks.append(
                        SimpleTrack(
                            track_id,
                            center,
                        )
                    )

            # Footfall analytics
            events = pipeline.update(tracks)

            # FPS
            fps_counter.update()
            frame_count += 1

            # Print events
            for event in events:

                print(
                    f"EVENT | "
                    f"Track ID={event['track_id']} | "
                    f"{event['event'].upper()}"
                )

            # Print statistics
            if frame_count % 100 == 0:

                summary = pipeline.summary()

                print()
                print(f"FRAME      : {frame_count}")
                print(f"FPS        : {fps_counter.fps:.2f}")
                print(f"ENTRIES    : {summary['entries']}")
                print(f"EXITS      : {summary['exits']}")
                print(f"OCCUPANCY  : {summary['occupancy']}")
                print()

    except KeyboardInterrupt:

        print("\nStopping...")

        summary = pipeline.summary()

        print()
        print("Final OpenVINO Footfall Summary")
        print("--------------------------------")
        print(f"Frames     : {frame_count}")
        print(f"FPS        : {fps_counter.fps:.2f}")
        print(f"Entries    : {summary['entries']}")
        print(f"Exits      : {summary['exits']}")
        print(f"Occupancy  : {summary['occupancy']}")

    finally:

        reader.release()


if __name__ == "__main__":
    main()
