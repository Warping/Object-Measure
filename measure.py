'''
    This script measures the dimensions of objects in a video stream using OpenCV.
    It detects a red reference object and calculates the dimensions of other objects
    in the frame based on the known dimensions of the reference object.
    
    Requirements:
    
    We recommend using Python 3.12 or later for this script.
    
    Create a virtual environment and install the required packages:
    
        python -m venv .venv
        .venv\Scripts\activate
        pip install -r requirements.txt
    
    Usage:
    
        python measure.py --video <path_to_video> --width <known_width>
        
    Example:
    
        python measure.py --video OneofEach.mp4 --width 1.0
        
'''

# Define known dimensions for accuracy check
washer_known = (0.7, 0.7)
screw_known = (0.2, 2.9)
hexnut_known = (0.4, 0.4)

# Minimum percent of frames required for accuracy report
min_percentage = 80

# Import necessary libraries
from scipy.spatial import distance as dist
from imutils import perspective
import numpy as np
import argparse
import imutils
import cv2

# Function definitions

# Function to calculate the midpoint between two points
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Function to filter out outliers based on mean and standard deviation
# This function removes points that are more than 2 standard deviations away from the mean
# It returns the filtered data
def filter_outliers(data):
    if len(data) == 0:
        return data
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    filtered = [d for d in data if np.all(np.abs(d - mean) < 2 * std)]
    return np.array(filtered)

# Function to calculate the error percentage between measured and known dimensions
# It returns a tuple of error percentages for both dimensions
# If the known dimension is zero, it returns (0, 0) to avoid division by zero
def calc_error(measured, known):
    if known[0] == 0 or known[1] == 0:
        return (0, 0)
    errA = 100 * abs(measured[0] - known[0]) / known[0]
    errB = 100 * abs(measured[1] - known[1]) / known[1]
    return (float(errA), float(errB))

# Main script execution

# Argument parser to handle command line arguments
# The script requires a video file and the known width of the reference object
# The known width is used to calculate the pixels per metric
# The default width is set to 1.0 inches
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
    help="path to the input video")
ap.add_argument("-w", "--width", type=float, default=1.0,
    help="known width of the red reference object in inches (default 1 inch)")
args = vars(ap.parse_args())

# Open the video file
cap = cv2.VideoCapture(args["video"])
if not cap.isOpened():
    print("[ERROR] Could not open video.")
    exit()

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[INFO] Total frames in video: {total_frames}")

# Frame counters
frames_with_washer = 0
frames_with_screw = 0
frames_with_hexnut = 0
total_processed_frames = 0

# Collect dimensions for analysis
washer_dims = []
screw_dims = []
hexnut_dims = []

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process the frame
    total_processed_frames += 1
    image = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Resize the image if it's too wide
    max_width = 700
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

    orig = image.copy()

    # Detect Red Reference Object
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    ref_cnts = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ref_cnts = imutils.grab_contours(ref_cnts)
    
    # Exit if no reference object is found
    if len(ref_cnts) == 0:
        continue

    # Get midpoint of the reference object
    ref_c = max(ref_cnts, key=cv2.contourArea)
    ref_box = cv2.minAreaRect(ref_c)
    ref_box = cv2.boxPoints(ref_box)
    ref_box = np.array(ref_box, dtype="int")
    ref_box = perspective.order_points(ref_box)

    (tl, tr, br, bl) = ref_box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    
    # Calculate pixel per metric ratio
    pixelsPerMetric = dB / args["width"]

    ref_center = ((tl[0] + br[0]) / 2, (tl[1] + br[1]) / 2)

    # Draw reference object boundary
    cv2.drawContours(orig, [ref_box.astype("int")], -1, (0, 0, 255), 2)
    cv2.putText(orig, "Reference object", (int(tl[0]), int(tl[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(orig, "{:.1f}in".format(args["width"]),
                (int(tr[0] + 10), int(tr[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
       
    # Find contours of other objects
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    washer_objects = []
    screw_objects = []
    hexnut_objects = []

    # Loop through contours and classify objects
    for c in cnts:
        # Ignore small contours
        if cv2.contourArea(c) < 100:
            continue
        # Ignore contours that are too similar to the reference object
        if cv2.matchShapes(c, ref_c, 1, 0.0) < 0.01:
            continue
        
        
        # Calculate dimensions
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        # Ignore small dimensions
        if dimA < 0.1 or dimB < 0.1:
            continue
        
        # Ignore objects that are too close to the reference object
        area = dimA * dimB
        distance_inches = dist.euclidean(ref_center, ((tl[0] + br[0]) / 2, (tl[1] + br[1]) / 2)) / pixelsPerMetric
        if distance_inches < 0.5:
            continue

        # Classify Hex nuts based on area and aspect ratio
        if area < (0.5 * 0.5):
            hexnut_objects.append({"box": box, "dimA": dimA, "dimB": dimB})
            
        # Classify Washers and Screws based on aspect ratio
        else:
            aspect = dimA / dimB if dimB != 0 else 0
            if aspect > 1.1 or aspect < 0.5:
                screw_objects.append({"box": box, "dimA": dimA, "dimB": dimB})
            else:
                washer_objects.append({"box": box, "dimA": dimA, "dimB": dimB})

    # Filter out overlapping hex nuts
    valid_hexnuts = []
    for hex_obj in hexnut_objects:
        hx_min = np.min(hex_obj["box"], axis=0)
        hx_max = np.max(hex_obj["box"], axis=0)
        overlap = False

        for obj_list in [washer_objects, screw_objects]:
            for other in obj_list:
                ox_min = np.min(other["box"], axis=0)
                ox_max = np.max(other["box"], axis=0)
                if (hx_min[0] < ox_max[0] and hx_max[0] > ox_min[0] and
                    hx_min[1] < ox_max[1] and hx_max[1] > ox_min[1]):
                    overlap = True
                    break
            if overlap:
                break

        if not overlap:
            valid_hexnuts.append(hex_obj)

    # Count the number of frames with each object
    
    # Washers
    if washer_objects:
        frames_with_washer += 1
        for obj in washer_objects:
            (tl, tr, br, bl) = obj["box"]
            dimA = obj["dimA"]
            dimB = obj["dimB"]
            washer_dims.append((dimA, dimB))
            cv2.drawContours(orig, [obj["box"].astype("int")], -1, (0, 255, 0), 2)
            cv2.putText(orig, "Washer", (int(tl[0]), int(tl[1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            # Display dimensions
            cv2.putText(orig, "{:.1f}in".format(dimA),
                        (int(tl[0] - 15), int(tl[1] - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(orig, "{:.1f}in".format(dimB),
                        (int(tr[0] + 10), int(tr[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Screws
    if screw_objects:
        frames_with_screw += 1
        for obj in screw_objects:
            (tl, tr, br, bl) = obj["box"]
            dimA = obj["dimA"]
            dimB = obj["dimB"]
            screw_dims.append((dimA, dimB))
            cv2.drawContours(orig, [obj["box"].astype("int")], -1, (0, 0, 255), 2)
            cv2.putText(orig, "Screw", (int(tl[0]), int(tl[1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            cv2.putText(orig, "{:.1f}in".format(dimA),
                        (int(tl[0] - 15), int(tl[1] - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(orig, "{:.1f}in".format(dimB),
                        (int(tr[0] + 10), int(tr[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Hex nuts
    if valid_hexnuts:
        frames_with_hexnut += 1
        for obj in valid_hexnuts:
            (tl, tr, br, bl) = obj["box"]
            dimA = obj["dimA"]
            dimB = obj["dimB"]
            hexnut_dims.append((dimA, dimB))
            cv2.drawContours(orig, [obj["box"].astype("int")], -1, (255, 255, 0), 2)
            cv2.putText(orig, "Hex nut", (int(tl[0]), int(tl[1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2.putText(orig, "{:.1f}in".format(dimA),
                        (int(tl[0] - 15), int(tl[1] - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(orig, "{:.1f}in".format(dimB),
                        (int(tr[0] + 10), int(tr[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the frame with measurements
    cv2.imshow("Measured Video", orig)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        while True:
            key2 = cv2.waitKey(0) & 0xFF
            if key2 == ord('p'):
                break
            elif key2 == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()
            elif key2 == ord('s'):
                filename = f"saved_frame_{total_processed_frames}.png"
                cv2.imwrite(filename, orig)
                print(f"[INFO] Saved paused frame as {filename}")

    elif key == ord('s'):
        filename = f"saved_frame_{total_processed_frames}.png"
        cv2.imwrite(filename, orig)
        print(f"[INFO] Saved current frame as {filename}")

cap.release()
cv2.destroyAllWindows()

# Perform final calculations and print results

# Filter outliers from dimensions
washer_dims = filter_outliers(np.array(washer_dims))
screw_dims = filter_outliers(np.array(screw_dims))
hexnut_dims = filter_outliers(np.array(hexnut_dims))

# Average dimensions
washer_avg = np.mean(washer_dims, axis=0) if len(washer_dims) > 0 else (0, 0)
screw_avg = np.mean(screw_dims, axis=0) if len(screw_dims) > 0 else (0, 0)
hexnut_avg = np.mean(hexnut_dims, axis=0) if len(hexnut_dims) > 0 else (0, 0)

# Calculate errors
washer_error = calc_error(washer_avg, washer_known)
screw_error = calc_error(screw_avg, screw_known)
hexnut_error = calc_error(hexnut_avg, hexnut_known)

# Output results
print("\n[RESULTS]")
print(f"Total frames processed: {total_processed_frames}")
print(f"Frames with Washer: {frames_with_washer} ({100 * frames_with_washer / total_processed_frames:.2f}%)")
print(f"Frames with Screw: {frames_with_screw} ({100 * frames_with_screw / total_processed_frames:.2f}%)")
print(f"Frames with Hex nut: {frames_with_hexnut} ({100 * frames_with_hexnut / total_processed_frames:.2f}%)")

print("\n--- Measurement Accuracy ---")

if 100 * frames_with_washer / total_processed_frames >= min_percentage:
    print(f"Washer avg dims: ({washer_avg[0]:.2f}, {washer_avg[1]:.2f}) | Known: {washer_known} | Error %: ({washer_error[0]:.2f}%, {washer_error[1]:.2f}%)")
else:
    print("Washer not detected. Skipping accuracy report.")

if 100 * frames_with_screw / total_processed_frames >= min_percentage:
    print(f"Screw avg dims: ({screw_avg[0]:.2f}, {screw_avg[1]:.2f}) | Known: {screw_known} | Error %: ({screw_error[0]:.2f}%, {screw_error[1]:.2f}%)")
else:
    print("Screw not detected. Skipping accuracy report.")

if 100 * frames_with_hexnut / total_processed_frames >= min_percentage:
    print(f"Hex nut avg dims: ({hexnut_avg[0]:.2f}, {hexnut_avg[1]:.2f}) | Known: {hexnut_known} | Error %: ({hexnut_error[0]:.2f}%, {hexnut_error[1]:.2f}%)")
else:
    print("Hex nut not detected. Skipping accuracy report.")

