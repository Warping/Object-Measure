import cv2
import numpy as np
import os

# ----------- PARAMETERS -----------
video_path = "Singlewasher.mp4"   # Replace with your video file
output_folder = "images/"  # Folder to save frames

# ----------- CREATE OUTPUT FOLDER IF NEEDED -----------
os.makedirs(output_folder, exist_ok=True)

# ----------- LOAD VIDEO -----------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Could not open video")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {frame_count}")

# ----------- SELECT 10 FRAME INDICES -----------
num_frames_to_extract = 10
frame_indices = np.linspace(0, frame_count - 1, num_frames_to_extract, dtype=int)

print(f"Frames to extract: {frame_indices}")

# ----------- EXTRACT, ROTATE, AND SAVE FRAMES -----------
extracted = 0
for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        # Rotate the frame 90 degrees clockwise
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        filename = f"{output_folder}frame_{extracted + 1}.jpg"
        cv2.imwrite(filename, rotated_frame)
        print(f"Saved: {filename}")
        extracted += 1
    else:
        print(f"Could not read frame at index {idx}")

cap.release()
print("Extraction and rotation completed.")
