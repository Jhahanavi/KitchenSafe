from edge_impulse_linux.image import ImageImpulseRunner


MODEL_PATHS = {
    "burner": "/home/arduino/models/burner-linux-aarch64-v2-impulse-#1.eim",
    "cloth": "/home/arduino/models/cloth-linux-aarch64-v1-impulse-#1.eim",
    "kitchen": "/home/arduino/models/kitchen-only-linux-aarch64-v1-impulse-#1.eim",
}


class KitchenSafeInference:
    """Loads and runs the three KitchenSafe Edge Impulse models."""

    def __init__(self):
        self.runners = {}

    def load_models(self):
        """Load all three EIM models."""
        for name, path in MODEL_PATHS.items():
            print(f"\nLoading {name.upper()}...")

            runner = ImageImpulseRunner(path)
            info = runner.init()

            self.runners[name] = runner

            print(f"✅ {name.upper()} loaded")
            print("Labels:", info["model_parameters"]["labels"])
            print(
                "Input:",
                info["model_parameters"]["image_input_width"],
                "x",
                info["model_parameters"]["image_input_height"],
            )

    def detect(self, frame):
        """
        Run all three models on one OpenCV frame.

        Returns:
            Dictionary containing detections from each model.
        """
        results = {}

        for name, runner in self.runners.items():

            # Convert the OpenCV frame into the model's input features.
            features, _ = runner.get_features_from_image(frame)

            # Run inference.
            result = runner.classify(features)

            results[name] = result

        return results

    def stop(self):
        """Stop all model runners."""
        for runner in self.runners.values():
            runner.stop()

        self.runners.clear()
