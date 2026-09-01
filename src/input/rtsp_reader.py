import cv2


class RTSPReader:
    def __init__(self, url):
        self.url = url
        self.cap = None

    def connect(self):
        print("Connecting to RTSP stream...")

        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open RTSP stream")

        print("RTSP stream connected.")

    def read(self):
        if self.cap is None:
            raise RuntimeError("RTSP stream is not connected")

        ret, frame = self.cap.read()

        return ret, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
