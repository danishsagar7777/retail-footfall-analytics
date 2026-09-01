import os
import yaml
import cv2

from src.input.rtsp_reader import RTSPReader
from src.input.frame_processor import resize_frame
from src.tracking.tracker import PersonTracker


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

    x1, y1, x2, y2 = counting_line

    # --------------------------------------------------
    # Get RTSP URL
    # --------------------------------------------------

    rtsp_url = os.getenv("RTSP_URL")

    if not rtsp_url:
        raise RuntimeError(
            "RTSP_URL is not set."
        )

    # --------------------------------------------------
    # Initialize
    # --------------------------------------------------

    reader = RTSPReader(rtsp_url)
    tracker = PersonTracker()

    print("Connecting to RTSP stream...")

    reader.connect()

    print("RTSP stream connected.")
    print("Starting visualization...")
    print("Press Q to quit.")

    try:

        while True:

            ret, frame = reader.read()

            if not ret:
                continue

            # Resize to 1280x720.
            frame = resize_frame(frame)

            # --------------------------------------------------
            # Track persons
            # --------------------------------------------------

            tracks = tracker.update(frame)

            # --------------------------------------------------
            # Draw light-violet counting line
            # --------------------------------------------------

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (211, 160, 211),
                4
            )

            # --------------------------------------------------
            # Draw tracks
            # --------------------------------------------------

            for track in tracks:

                bx1, by1, bx2, by2 = map(
                    int,
                    track.bbox
                )

                cx, cy = map(
                    int,
                    track.center
                )

                # Bounding box
                cv2.rectangle(
                    frame,
                    (bx1, by1),
                    (bx2, by2),
                    (0, 255, 0),
                    2
                )

                # Center point
                cv2.circle(
                    frame,
                    (cx, cy),
                    5,
                    (0, 0, 255),
                    -1
                )

                # Track ID
                cv2.putText(
                    frame,
                    f"ID: {track.track_id}",
                    (bx1, max(20, by1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            # --------------------------------------------------
            # Display
            # --------------------------------------------------

            cv2.imshow(
                "Retail Footfall Tracking",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except KeyboardInterrupt:

        print("\nStopping...")

    finally:

        reader.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
