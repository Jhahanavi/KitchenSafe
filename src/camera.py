import cv2


CAMERA_DEVICE = "/dev/video2"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def open_camera():
    """Open the C270 USB webcam using V4L2."""
    print(f"Starting camera: {CAMERA_DEVICE}")

    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {CAMERA_DEVICE}"
        )

    # Request the same resolution used during our successful test.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("✅ Camera opened")

    return cap


def read_frame(cap):
    """Capture one frame from the camera."""
    ret, frame = cap.read()

    if not ret:
        raise RuntimeError("Could not capture frame")

    return frame


def close_camera(cap):
    """Release the camera."""
    if cap is not None:
        cap.release()
