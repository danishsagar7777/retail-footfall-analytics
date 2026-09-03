import cv2


def resize_frame(frame, width=640, height=640):
    """
    Resize a frame to the inference resolution.

    The current OpenVINO YOLO model expects
    a fixed input shape of 640x640.
    """
    return cv2.resize(
        frame,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
