import cv2


def resize_frame(frame, width=1280, height=720):
    return cv2.resize(
        frame,
        (width, height),
        interpolation=cv2.INTER_AREA
    )
