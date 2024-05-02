import os
import cv2
import mediapipe as mp
import csv

def process_video(video_path, csv_writer, condition_severity, pose, frame_id_start=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return frame_id_start  # If file can't be opened, return and do not increment frame_id

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video {video_path.split(os.sep)[-1]} with {total_frames} frames.")

    frame_id = frame_id_start
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        keypoints = [[landmark.x, landmark.y] if landmark.visibility > 0.5 else (0, 0) for landmark in results.pose_landmarks.landmark] if results.pose_landmarks else [[0, 0] for _ in range(33)]
        row = [frame_id] + [coord for point in keypoints for coord in (point[0], point[1])] + [condition_severity]
        csv_writer.writerow(row)
        print(f"Processed frame {frame_id+1}/{total_frames}", end='\r')
        frame_id += 1
    print("\nFinished processing video.")
    cap.release()
    return frame_id

def process_directory_structure(data_directory):
    mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    total_videos = sum([len(files) for r, d, files in os.walk(data_directory) if any(file.endswith('.MOV') for file in files)])
    processed_videos = 0

    for root, dirs, files in os.walk(data_directory):
        if files:
            metadata_parts = root.split(os.sep)[-1].split('_')
            condition = metadata_parts[0]
            severity = metadata_parts[1] if len(metadata_parts) > 1 else 'NA'
            condition_severity = f'{condition}_{severity}'
            csv_filename = f'{condition}_{severity}_keypoints.csv'
            csv_path = os.path.join(root, csv_filename)
            with open(csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                header = ['frame_id'] + [f'point_{i}x, point_{i}_y' for i in range(33)] + ['target']
                writer.writerow(header)
                frame_id = 0
                for filename in files:
                    if filename.endswith('.MOV'):
                        video_path = os.path.join(root, filename)
                        frame_id = process_video(video_path, writer, condition_severity, mp_pose, frame_id)
                        processed_videos += 1
                        print(f"Completed processing {video_path}. ({processed_videos}/{total_videos} videos processed)")

    mp_pose.close()

# Specify the correct directory path
data_directory = r'Data\KOA-PD-NM'
process_directory_structure(data_directory)
