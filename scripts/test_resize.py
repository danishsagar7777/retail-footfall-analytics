import os
import cv2

from src.input.rtsp_reader import RTSPReader
from src.input.frame_processor import resize_frame


def main():
    rtsp_url = os.getenv("RTSP_URL")

    if not rtsp_url:
        raise RuntimeError("RTSP_URL is not set.")

    reader = RTSPReader(rtsp_url)
    reader.connect()

    ret, frame = reader.read()

    if not ret:
        raise RuntimeError("Could not read frame.")

    print("Original:", frame.shape)

    resized = resize_frame(frame)

    print("Resized:", resized.shape)

    reader.release()


if __name__ == "__main__":
    main()
