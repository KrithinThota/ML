import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # To disable XNNPACK delegate for CPU inference

import cv2
import mediapipe as mp
import csv

# Load the MediaPipe pose detection model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the video file for reading
cap = cv2.VideoCapture(r'Data\testdata.mp4') 
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Open a CSV file for writing keypoints
with open('keypoints.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ['frame_id'] + [f'point_{i}x, point{i}_y' for i in range(33)]  
    writer.writerow(header)
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB (MediaPipe requires RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect poses in the frame
        results = pose.process(rgb_frame)
        if results.pose_landmarks is not None:
            keypoints = [[landmark.x, landmark.y] for landmark in results.pose_landmarks.landmark]
        else:
            keypoints = [[0, 0] for _ in range(33)]  # If no keypoints detected, fill with zeros
        
        # Write keypoints to the CSV file
        row = [frame_id] + [coord for point in keypoints for coord in (point[0], point[1])]
        writer.writerow(row)
        
        frame_id += 1

cap.release()

pose.close()
