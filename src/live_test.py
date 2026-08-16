import cv2
import subprocess
import time
from edge_impulse_linux.image import ImageImpulseRunner

CAMERA = "/dev/video2"

MODELS = {
    "BURNER": "/home/arduino/models/burner-linux-aarch64-v2-impulse-#1.eim",
    "CLOTH": "/home/arduino/models/cloth-linux-aarch64-v1-impulse-#1.eim",
    "KITCHEN": "/home/arduino/models/kitchen-only-linux-aarch64-v1-impulse-#1.eim"
}

# --------------------------------------------------
# Start GStreamer camera pipeline
# --------------------------------------------------

gst_command = [
    "gst-launch-1.0",
    "v4l2src",
    f"device={CAMERA}",
    "!",
    "image/jpeg,width=640,height=480,framerate=30/1",
    "!",
    "jpegdec",
    "!",
    "videoconvert",
    "!",
    "jpegenc",
    "!",
    "multifilesink",
    "location=/tmp/kitchensafe_live.jpg",
    "post-messages=true"
]

print("Starting C270 camera...")
print("Camera:", CAMERA)

# --------------------------------------------------
# Load models
# --------------------------------------------------

runners = {}

for name, model in MODELS.items():

    print(f"\nLoading {name}...")

    runner = ImageImpulseRunner(model)
    info = runner.init()

    runners[name] = runner

    print(f"✅ {name} loaded")
    print("Labels:", info["model_parameters"]["labels"])

print("\n======================================")
print("KITCHENSAFE LIVE 3-MODEL TEST")
print("======================================")
print("Camera:", CAMERA)
print("Press Ctrl+C to stop")
print("======================================")

# --------------------------------------------------
# Camera process
# --------------------------------------------------

process = subprocess.Popen(
    gst_command,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

last_mtime = 0

try:

    while True:

        # Wait for a new camera frame
        try:
            mtime = __import__("os").path.getmtime(
                "/tmp/kitchensafe_live.jpg"
            )
        except FileNotFoundError:
            time.sleep(0.1)
            continue

        if mtime == last_mtime:
            time.sleep(0.05)
            continue

        last_mtime = mtime

        # Read frame
        frame = cv2.imread("/tmp/kitchensafe_live.jpg")

        if frame is None:
            continue

        print("\n========== NEW FRAME ==========")

        # Run all three models
        for name, runner in runners.items():

            features, _ = runner.get_features_from_image(frame)

            result = runner.classify(features)

            boxes = result.get("result", {}).get(
                "bounding_boxes", []
            )

            print(f"\n{name}:")

            if not boxes:
                print("  No detections")
            else:

                for box in boxes:

                    label = box.get("label")
                    confidence = box.get("value", 0)

                    x = box.get("x")
                    y = box.get("y")
                    w = box.get("width")
                    h = box.get("height")

                    print(
                        f"  {label} "
                        f"{confidence:.2f} "
                        f"bbox=({x},{y},{w},{h})"
                    )

        # Don't run inference at maximum camera FPS
        time.sleep(0.3)

except KeyboardInterrupt:

    print("\nStopping...")

finally:

    process.terminate()

    for runner in runners.values():
        runner.stop()

    print("All models stopped.")
