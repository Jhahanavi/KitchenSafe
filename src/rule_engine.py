class KitchenSafeRuleEngine:
    """
    Converts detections from the three AI models into a KitchenSafe
    safety state.

    No IoU is used.
    """

    CONFIDENCE_THRESHOLD = 0.50

    def process(self, results):
        """
        Process the output from all three models.

        Returns a dictionary containing:
            - detected conditions
            - overall safety state
            - reason
        """

        burner_detections = self._get_boxes(
            results.get("burner", {})
        )

        cloth_detections = self._get_boxes(
            results.get("cloth", {})
        )

        kitchen_detections = self._get_boxes(
            results.get("kitchen", {})
        )

        # -----------------------------------------
        # BURNER MODEL
        # -----------------------------------------

        burner_on = any(
            self._label(box) == "1"
            and self._confidence(box) >= self.CONFIDENCE_THRESHOLD
            for box in burner_detections
        )

        burner_off = any(
            self._label(box) == "0"
            and self._confidence(box) >= self.CONFIDENCE_THRESHOLD
            for box in burner_detections
        )

        # -----------------------------------------
        # CLOTH MODEL
        # -----------------------------------------

        cloth_detected = any(
            self._confidence(box) >= self.CONFIDENCE_THRESHOLD
            for box in cloth_detections
        )

        # -----------------------------------------
        # KITCHEN MODEL
        # -----------------------------------------

        flame_detected = self._has_kitchen_class(
            kitchen_detections, "0"
        )

        smoke_detected = self._has_kitchen_class(
            kitchen_detections, "1"
        )

        kitchen_burner_detected = self._has_kitchen_class(
            kitchen_detections, "2"
        )

        spill_detected = self._has_kitchen_class(
            kitchen_detections, "3"
        )

        safe_detected = self._has_kitchen_class(
            kitchen_detections, "4"
        )

        # -----------------------------------------
        # SAFETY DECISION
        # -----------------------------------------

        state = "SAFE"
        reason = "No kitchen hazard detected."

        if flame_detected or smoke_detected:
            state = "DANGER"

            reasons = []

            if flame_detected:
                reasons.append("Flame detected")

            if smoke_detected:
                reasons.append("Smoke detected")

            reason = " + ".join(reasons)

        elif burner_on and cloth_detected:
            state = "DANGER"
            reason = "Cloth detected while burner is ON."

        elif spill_detected:
            state = "WARNING"
            reason = "Spill detected."

        elif burner_on:
            state = "WARNING"
            reason = "Burner is ON."

        elif cloth_detected:
            state = "WARNING"
            reason = "Cloth detected."

        return {
            "state": state,
            "reason": reason,
            "detections": {
                "burner_on": burner_on,
                "burner_off": burner_off,
                "cloth": cloth_detected,
                "flame": flame_detected,
                "smoke": smoke_detected,
                "kitchen_burner": kitchen_burner_detected,
                "spill": spill_detected,
                "safe": safe_detected,
            },
        }

    # -----------------------------------------
    # Helper functions
    # -----------------------------------------

    def _get_boxes(self, result):
        """Extract bounding boxes from an Edge Impulse result."""

        if not result:
            return []

        return result.get("result", {}).get("bounding_boxes", [])

    def _label(self, box):
        """Get the class label from a bounding box."""

        return str(
            box.get(
                "label",
                box.get("class_name", "")
            )
        )

    def _confidence(self, box):
        """Get confidence score from a bounding box."""

        return float(
            box.get(
                "value",
                box.get("confidence", 0.0)
            )
        )

    def _has_kitchen_class(self, boxes, class_id):
        """Check whether a Kitchen model class is detected."""

        return any(
            self._label(box) == class_id
            and self._confidence(box) >= self.CONFIDENCE_THRESHOLD
            for box in boxes
        )
