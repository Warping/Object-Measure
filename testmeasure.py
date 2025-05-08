from scipy.spatial import distance as dist
from imutils import perspective
import numpy as np
import argparse
import imutils
import cv2
import os

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def filter_outliers(data):
    if len(data) == 0:
        return data
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    filtered = [d for d in data if np.all(np.abs(d - mean) < 2 * std)]
    return np.array(filtered)

def calc_error(measured, known):
    if known[0] == 0 or known[1] == 0:
        return (0, 0)
    errA = 100 * abs(measured[0] - known[0]) / known[0]
    errB = 100 * abs(measured[1] - known[1]) / known[1]
    return (float(errA), float(errB))

# --- Arguments ---
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
    help="path to the input video")
ap.add_argument("-w", "--width", type=float, default=1.0,
    help="known width of the red reference object in inches (default 1 inch)")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["video"])
if not cap.isOpened():
    print("[ERROR] Could not open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[INFO] Total frames in video: {total_frames}")

# Create folder for saved frames
if not os.path.exists("SavedFrames"):
    os.makedirs("SavedFrames")

# ------------------ Determine proper output size -------------------
ret, frame = cap.read()
if not ret:
    print("[ERROR] Could not read the first frame.")
    exit()

image = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
max_width = 700
height, width = image.shape[:2]
if width > max_width:
    scale = max_width / width
    image = cv2.resize(image, (int(width * scale), int(height * scale)))

output_height, output_width = image.shape[:2]
output_size = (output_width, output_height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("processed_output.mp4", fourcc, fps, output_size)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# ------------------------------------------------------------------------

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
    ret, frame = cap.read()
    if not ret:
        break

    total_processed_frames += 1
    image = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

    orig = image.copy()

    # --- Step 1: Detect Red Reference ---
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
    if len(ref_cnts) == 0:
        continue

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
    pixelsPerMetric = dB / args["width"]

    ref_center = ((tl[0] + br[0]) / 2, (tl[1] + br[1]) / 2)

    cv2.drawContours(orig, [ref_box.astype("int")], -1, (0, 0, 255), 2)
    cv2.putText(orig, "Reference object", (int(tl[0]), int(tl[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # --- Edge Detection ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # --- Object Detection ---
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    washer_objects = []
    screw_objects = []
    hexnut_objects = []

    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue
        if cv2.matchShapes(c, ref_c, 1, 0.0) < 0.01:
            continue

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

        if dimA < 0.1 or dimB < 0.1:
            continue

        area = dimA * dimB
        distance_inches = dist.euclidean(ref_center, ((tl[0] + br[0]) / 2, (tl[1] + br[1]) / 2)) / pixelsPerMetric
        if distance_inches < 0.5:
            continue

        if area < (0.5 * 0.5):
            hexnut_objects.append({"box": box, "dimA": dimA, "dimB": dimB})
        else:
            aspect = dimA / dimB if dimB != 0 else 0
            if aspect > 1.1 or aspect < 0.5:
                screw_objects.append({"box": box, "dimA": dimA, "dimB": dimB})
            else:
                washer_objects.append({"box": box, "dimA": dimA, "dimB": dimB})

    # --- Draw Detected Objects ---
    def draw_objects(objects, label, color):
        for obj in objects:
            (tl, tr, br, bl) = obj["box"]
            dimA, dimB = obj["dimA"], obj["dimB"]
            cv2.drawContours(orig, [obj["box"].astype("int")], -1, color, 2)
            cv2.putText(orig, label, (int(tl[0]), int(tl[1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(orig, f"{dimA:.1f}in", (int(tl[0] - 15), int(tl[1] - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(orig, f"{dimB:.1f}in", (int(tr[0] + 10), int(tr[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    draw_objects(washer_objects, "Washer", (0, 255, 0))
    draw_objects(screw_objects, "Screw", (0, 0, 255))
    draw_objects(hexnut_objects, "Hex nut", (255, 255, 0))

    # --- Write frame to video ---
    out.write(orig)

    # --- Show Previews ---
    preview_scale = 0.5
    cv2.imshow("Original", cv2.resize(image, None, fx=preview_scale, fy=preview_scale))
    cv2.imshow("Red Mask", cv2.resize(red_mask, None, fx=preview_scale, fy=preview_scale))
    cv2.imshow("Edges", cv2.resize(edged, None, fx=preview_scale, fy=preview_scale))
    cv2.imshow("Measured Video", cv2.resize(orig, None, fx=preview_scale, fy=preview_scale))

    # --- Controls ---
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
                out.release()
                cv2.destroyAllWindows()
                exit()
            elif key2 == ord('s'):
                filename = f"SavedFrames/frame_{total_processed_frames}.png"
                cv2.imwrite(filename, orig)
                print(f"[INFO] Saved paused frame as {filename}")
    elif key == ord('s'):
        filename = f"SavedFrames/frame_{total_processed_frames}.png"
        cv2.imwrite(filename, orig)
        print(f"[INFO] Saved frame as {filename}")

cap.release()
out.release()
cv2.destroyAllWindows()
