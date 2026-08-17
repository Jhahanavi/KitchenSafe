# KitchenSafe

AI-powered kitchen safety and hazard detection system using an Arduino UNO Q, Edge Impulse models, and a USB camera.

## Overview

KitchenSafe is designed to detect potentially dangerous situations in a kitchen using computer vision and multiple Edge Impulse object-detection models.

The system uses three models simultaneously:

1. **Burner Model**
   - `0` → Burner OFF
   - `1` → Burner ON

2. **Cloth Model**
   - Detects cloth objects.
   - The class ID is not important for the safety logic; any valid cloth detection is treated as cloth.

3. **Kitchen Model**
   - `0` → Flame
   - `1` → Smoke
   - `2` → Burner
   - `3` → Spill
   - `4` → Safe

The detections from these models are combined by a safety rule engine to determine the overall kitchen safety state.

---

## Hardware

- Arduino UNO Q
- C270 USB Webcam
- USB connection for the webcam

### Camera

The C270 webcam is currently detected on the Arduino UNO Q as:

```text
/dev/video2
