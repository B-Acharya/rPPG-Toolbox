import cv2
import numpy as np

# Path to the video file
video_path = "/data/sit_22_alu_video_mix_lab/processed/hrv/pre04179_part_2_disgust.mkv"

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

frame_count = 0

# Read and process frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Break the loop when no frames are left
    if not ret:
        print("End of video or cannot read the frame.")
        break

    # Convert the frame to a numpy array
    frame_array = np.array(frame)

    # Print the numpy array of the frame
    print(f"Frame {frame_count}:\n", frame_array)

    # Increment frame counter
    frame_count += 1

# Release the video capture object
cap.release()

print("Video processing completed.")
