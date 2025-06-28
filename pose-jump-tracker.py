import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture("test_videos/jumprope.mp4")
jump_count = 0
prev_y = None
jumping = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        y = ankle.y

        # Simple jump detection
        if prev_y is not None:
            if y < prev_y - 0.05 and not jumping:
                jump_count += 1
                jumping = True
            elif y >= prev_y:
                jumping = False

        prev_y = y
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show jump count
    cv2.putText(frame, f"Jumps: {jump_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
    cv2.imshow("Jump Rope Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
