from camera import open_camera, read_frame, close_camera
from inference import KitchenSafeInference
from rule_engine import KitchenSafeRuleEngine


def main():
    print("\n======================================")
    print("        KITCHENSAFE")
    print("======================================")

    camera = None
    inference = KitchenSafeInference()
    rule_engine = KitchenSafeRuleEngine()

    try:
        # -----------------------------------------
        # Start camera
        # -----------------------------------------
        camera = open_camera()

        # -----------------------------------------
        # Load all three AI models
        # -----------------------------------------
        inference.load_models()

        print("\n======================================")
        print("KitchenSafe is running")
        print("Press Ctrl+C to stop")
        print("======================================")

        # -----------------------------------------
        # Real-time inference loop
        # -----------------------------------------
        while True:

            frame = read_frame(camera)

            # Run Burner + Cloth + Kitchen models
            results = inference.detect(frame)

            # Process AI results through safety rules
            safety = rule_engine.process(results)

            print("\n======================================")
            print("KITCHENSAFE STATUS")
            print("======================================")

            print(f"State  : {safety['state']}")
            print(f"Reason : {safety['reason']}")

            detections = safety["detections"]

            print("\nDetections:")
            print(f"  Burner ON       : {detections['burner_on']}")
            print(f"  Burner OFF      : {detections['burner_off']}")
            print(f"  Cloth           : {detections['cloth']}")
            print(f"  Flame           : {detections['flame']}")
            print(f"  Smoke           : {detections['smoke']}")
            print(f"  Kitchen Burner  : {detections['kitchen_burner']}")
            print(f"  Spill           : {detections['spill']}")
            print(f"  Safe            : {detections['safe']}")

    except KeyboardInterrupt:
        print("\n\nStopping KitchenSafe...")

    except Exception as e:
        print(f"\n❌ KitchenSafe error: {e}")

    finally:
        print("Cleaning up...")

        if camera is not None:
            close_camera(camera)

        inference.stop()

        print("✅ KitchenSafe stopped")


if __name__ == "__main__":
    main()
