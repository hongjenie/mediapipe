import cv2
import mediapipe as mp
import numpy as np
from IPython.display import clear_output

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)
counter = 0
stage = None

def calculate_angle(a, b, c):
    
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

## setup mediapipe instance
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB                  
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = True                

        # Make detection
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        
        # try:
        landmarks = results.pose_landmarks.landmark
        # Get coordinates
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        # Calculate angle
        angle = calculate_angle(knee, elbow, wrist)
        print(angle)
        
        # Visualize angle
        cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.rectangle(image, (0, 0), (250, 250), (245, 150, 25), 2)
        
        # Curl counter logic

        if angle > 120:
            stage = "down"
        
        if (angle < 50) and (stage =='down'):
            stage = "up"
            counter += 1
            print(counter)
        # except:
            # print("error raised")
            # pass
        # Render curl counter
       
        # Setup status box
        cv2.rectangle(image, (0, 0), (250, 250), (245, 150, 25), 2)
       
        # COUNT data
        cv2.putText(image, 'COUNT', (50, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color = (240, 117, 66), thickness = 2, circle_radius = 2),
        mp_drawing.DrawingSpec(color = (240, 66, 230), thickness = 2, circle_radius = 2)  
        )
        # print(results)
        # print(len(landmarks))
  
        cv2.imshow('Mediapipe Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        cap.release()
        cv2.destroyAllWindows()















































