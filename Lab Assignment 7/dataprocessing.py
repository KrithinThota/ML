import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
import tensorflow as tf
import csv


# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='code\dataset_pre_processing\movenet_singlepose_lightning.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess the image
def preprocess_image(image, input_size):
    img = tf.image.resize_with_pad(image, input_size, input_size)
    return img[tf.newaxis, ...]

# Open the video
cap = cv2.VideoCapture('Data\testdata.mp4') 
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up CSV file for storing keypoints
with open('keypoints.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    header = ['frame_id'] + [f'point_{i}x, point{i}_y' for i in range(17)]
    writer.writerow(header)
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the image
        input_image = preprocess_image(frame, input_details[0]['shape'][1])
        interpreter.set_tensor(input_details[0]['index'], input_image)

        # Run inference
        interpreter.invoke()

        # Extract keypoints
        keypoints = interpreter.get_tensor(output_details[0]['index'])[0][0]
        keypoints = keypoints[:, :2]  # Only x, y coordinates
        
        # Write keypoints to CSV
        row = [frame_id] + list(keypoints.flatten())
        writer.writerow(row)
        
        frame_id += 1

# Release the video and close files
cap.release()