# IOT_Driven_WildFire_Detection_System

# 🔥 IoT-Driven Real-Time Wildfire Detection

An efficient, lightweight wildfire detection system that uses a Raspberry Pi with a camera to detect fire in real-time using a TensorFlow Lite model and computer vision techniques. The system overlays a grid when fire is detected and provides visual feedback with detection confidence and FPS metrics.

---

## 🚀 Features

- 🔍 Real-time fire detection using a TFLite model  
- 🌈 Color-based filtering (HSV) to enhance model accuracy  
- 🧠 Grid overlay for fire localization  
- ⚡ Optimized for low-power edge devices (e.g. Raspberry Pi)  
- 🎯 Confidence thresholding to reduce false positives  
- 📊 Live FPS monitoring for performance insights  

---

## 🧰 Hardware & Software Requirements

### Hardware
- Raspberry Pi 4 (or 3B+)
- Raspberry Pi Camera Module v1.3+ or USB Webcam
- MicroSD Card (16GB+ recommended)
- Power supply

### Software
- Raspberry Pi OS (Bullseye recommended)
- Python 3
- TensorFlow Lite Runtime
- OpenCV
- NumPy

---

## 📦 Installation

```bash
# Update and install OpenCV
sudo apt update
sudo apt install python3-opencv python3-pip

# Install Python dependencies
pip3 install numpy tflite-runtime
```

If using a USB webcam, ensure `/dev/video0` is accessible and supported.

---

## 📁 Project Structure

```
wildfire-detection/
│
├── wildfire_compatible.tflite      # Pre-trained TFLite model
├── fire_detection.py               # Main detection script
├── README.md                       # Documentation
```

---

## ▶️ Usage

To start the real-time detection:

```bash
python3 fire_detection.py
```

**Controls:**  
Press `q` to quit the video stream.

---

## 🔍 Detection Logic

This system uses a hybrid approach combining:

- **Deep Learning (TFLite):** Classifies whether fire is present in each frame.
- **Color-Based Filtering:** Detects flame-like colors using HSV masking.
- **Stability Filter:** Requires 5 consistent fire-positive frames to trigger an alert.

**Fire is detected when:**
- TFLite model's confidence > 0.85
- Flame-like colors detected in frame
- Condition holds for 5+ consecutive frames

---

## 📸 Output

**When fire is detected:**
- 🔴 A red **"🔥 FIRE DETECTED!"** label appears.
- 🧱 A 3×3 red grid overlays the frame.
- 🔢 Score and FPS are shown live.

**When no fire is detected:**
- ✅ **"No Fire"** is displayed in green.

### 📊 Sample Output Frame

```
-------------------------------------
| 🔥 FIRE DETECTED!                 |
| Score: 0.96    FPS: 25.4          |
| ┌────┬────┬────┐                  |
| │    │    │    │  (3×3 Grid)      |
| ├────┼────┼────┤                  |
| │    │ 🔥 │    │                  |
| └────┴────┴────┘                  |
-------------------------------------
```

---

## 🛠️ Future Enhancements

- 🔔 Add physical buzzer or alarm using GPIO
- 🌐 Integrate with MQTT, Firebase, or email/SMS alerts
- 🧠 Train a custom fire dataset for better precision
- 💾 Capture and save snapshots when fire is detected
- 🛰️ Add GPS, temperature/humidity, and smoke sensors for a full IoT solution