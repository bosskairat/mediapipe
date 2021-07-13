import cv2
import time
import mediapipe as mp
import datetime


mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=True)
mp_draw = mp.solutions.drawing_utils

def holistic_recursive(origin_image, image, padding = 20, recursion_depth = 20):
    results = holistic.process(image)
    X , Y = [], []
    h, w = image.shape[:2]
    person_detected = False
    if results.pose_landmarks:
        mp_draw.draw_landmarks(origin_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        for i in range(len(results.pose_landmarks.landmark)):
            X.append(results.pose_landmarks.landmark[i].x)
            Y.append(results.pose_landmarks.landmark[i].y)
        person_detected = True
    if results.face_landmarks:    
        mp_draw.draw_landmarks(origin_image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        for i in range(len(results.face_landmarks.landmark)):
            X.append(results.face_landmarks.landmark[i].x)
            Y.append(results.face_landmarks.landmark[i].y)
        person_detected = True
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(origin_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        for i in range(len(results.left_hand_landmarks.landmark)):
            X.append(results.left_hand_landmarks.landmark[i].x)
            Y.append(results.left_hand_landmarks.landmark[i].y)
        person_detected = True
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(origin_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        for i in range(len(results.right_hand_landmarks.landmark)):
            X.append(results.right_hand_landmarks.landmark[i].x)
            Y.append(results.right_hand_landmarks.landmark[i].y)
        person_detected = True
    if person_detected and recursion_depth > 0:
        # create bbox
        x1 = int(min(X) * w)
        y1 = int(min(Y) * h)
        x2 = int(max(X) * w)
        y2 = int(max(Y) * h)
        # Add padding
        x1 = x1 - padding if x1 - padding > 0 else 0
        y1 = y1 - padding if y1 - padding > 0 else 0
        x2 = x2 + padding if x2 + padding < w else w
        y2 = y2 + padding if y2 + padding < h else h
        # zero bbox
        image[y1:y2, x1:x2] = 0
        # cv2.imwrite('crop.jpg', image)
        holistic_recursive(origin_image, image, recursion_depth - 1)

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('smeny-etsn-brigadoy-tkrs_Pxv28bmL_N5en.mp4')

# Write video
w = int(cap.get(3))
h = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output_recursive.mp4', fourcc, 10.0, (w, h))

start_time = time.time()
frame_count = 0
fps = 0
while True:
    _, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Recursive run detection
    image = img_rgb.copy()
    holistic_recursive(img, image)
    frame_count += 1
    if frame_count > 20:
        fps = frame_count / (time.time() - start_time)  
        start_time = time.time()
        frame_count = 0
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow("cam", img)
    out.write(img)
    k = cv2.waitKey(1)
    if k == 27:  # close on ESC key
        break

cap.release()
out.release()
cv2.destroyAllWindows()