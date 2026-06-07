import cv2
import requests
import time

FLASK_URL = "http://127.0.0.1:7000/face-auth"
NODE_URL = "http://127.0.0.1:3000/api/intruder-alert"
DEVICE_ID = "home-01"
CHECK_INTERVAL = 3

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

print("Camera is running... Press Q to quit.")

last_check = 0
overlay = []

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 6, minSize=(100, 100))

    for (x, y, w, h) in detected_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(detected_faces) > 0 and (time.time() - last_check) > CHECK_INTERVAL:
        last_check = time.time()
        overlay = []

        _, buffer = cv2.imencode('.jpg', frame)
        files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}

        try:
            response = requests.post(FLASK_URL, files=files, timeout=5).json()
            results = response.get('results', [])

            for i, result in enumerate(results):
                if result.get('known'):
                    name = result.get('name', 'Friend')
                    confidence = result.get('confidence', 0)
                    label = f"Welcome {name} ({confidence})"
                    color = (0, 255, 0)
                    print(f"Face {i+1}: Known - {name} | confidence: {confidence}")
                else:
                    confidence = result.get('confidence', 0)
                    label = f"INTRUDER! ({confidence})"
                    color = (0, 0, 255)
                    print(f"Face {i+1}: UNKNOWN | confidence: {confidence}")

                    # Send intruder alert to Node.js
                    try:
                        requests.post(NODE_URL, json={
                            "deviceId": DEVICE_ID,
                            "confidence": confidence
                        }, timeout=3)
                    except Exception as alert_err:
                        print(f"Alert error: {alert_err}")

                overlay.append((label, color))

        except Exception as e:
            print(f"Request error: {e}")

    # Draw overlay labels on frame
    for idx, (label, color) in enumerate(overlay):
        cv2.putText(frame, label, (10, 50 + idx * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Smart Home Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()