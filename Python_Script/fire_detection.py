import cv2
import numpy as np
import tensorflow.lite as tflite
import time

# Load TFLite model
interpreter = tflite.Interpreter(model_path="wildfire_compatible.tflite")
interpreter.allocate_tensors()

# Model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def run_inference(image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

def has_fire_like_colors(frame, threshold=500):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = (0, 100, 100)
    upper_red1 = (10, 255, 255)
    lower_red2 = (160, 100, 100)
    upper_red2 = (179, 255, 255)
    lower_yellow = (18, 100, 100)
    upper_yellow = (30, 255, 255)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = mask1 | mask2 | mask3
    return cv2.countNonZero(mask) > threshold

def draw_grid(frame, grid_size=(3, 3), color=(0, 0, 255), thickness=1):
    h, w = frame.shape[:2]
    dx = w // grid_size[0]
    dy = h // grid_size[1]
    for x in range(1, grid_size[0]):
        cv2.line(frame, (x * dx, 0), (x * dx, h), color, thickness)
    for y in range(1, grid_size[1]):
        cv2.line(frame, (0, y * dy), (w, y * dy), color, thickness)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution = smoother video
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Could not access camera.")
        return

    print("🔥 Fire detection started (TFLite). Press 'q' to quit.")

    fire_count = 0
    FIRE_FRAME_THRESHOLD = 5
    CONFIDENCE_THRESHOLD = 0.85
    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame capture failed.")
            break

        # Fire detection logic
        input_data = preprocess_image(frame)
        result = run_inference(input_data)
        score = result[0][0]
        is_confident = score > CONFIDENCE_THRESHOLD
        has_colors = has_fire_like_colors(frame)

        if is_confident and has_colors:
            fire_count += 1
        else:
            fire_count = max(0, fire_count - 1)

        if fire_count >= FIRE_FRAME_THRESHOLD:
            label = "🔥 FIRE DETECTED!"
            color = (0, 0, 255)
            draw_grid(frame)
        else:
            label = "✅ No Fire"
            color = (0, 255, 0)

        # FPS calculation
        frame_count += 1
        current_time = time.time()
        elapsed = current_time - prev_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            prev_time = current_time
        else:
            fps = None

        # Draw label and FPS
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Score: {score:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("🔥 Real-time Fire Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
