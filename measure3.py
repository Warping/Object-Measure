from scipy.spatial import distance as dist
from imutils import perspective
import numpy as np
import argparse
import imutils
import cv2

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# --- Arguments ---
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
    help="path to the input video")
ap.add_argument("-w", "--width", type=float, default=1.0,
    help="known width of the red reference object in inches (default 1 inch)")
args = vars(ap.parse_args())

# --- Video capture ---
cap = cv2.VideoCapture(args["video"])

if not cap.isOpened():
    print("[ERROR] Could not open video.")
    exit()

# --- Get total number of frames ---
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[INFO] Total frames in video: {total_frames}")

# --- Counters ---
frames_with_washer = 0
frames_with_screw = 0
frames_with_hexnut = 0
total_processed_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    total_processed_frames += 1

    # --- Rotate frame ---
    image = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # --- Resize BEFORE processing ---
    max_width = 1000
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
        continue  # No reference found

    ref_c = max(ref_cnts, key=cv2.contourArea)
    ref_box = cv2.minAreaRect(ref_c)
    ref_box = cv2.boxPoints(ref_box) if not imutils.is_cv2() else cv2.cv.BoxPoints(ref_box)
    ref_box = np.array(ref_box, dtype="int")
    ref_box = perspective.order_points(ref_box)

    (tl, tr, br, bl) = ref_box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    pixelsPerMetric = dB / args["width"]

    ref_centerX = (tl[0] + br[0]) / 2
    ref_centerY = (tl[1] + br[1]) / 2
    ref_center = (ref_centerX, ref_centerY)

    cv2.drawContours(orig, [ref_box.astype("int")], -1, (0, 0, 255), 2)

    cv2.putText(orig, "Reference object", (int(tl[0]), int(tl[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    cv2.putText(orig, "{:.1f}in".format(args["width"]),
                (int(tr[0] + 10), int(tr[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # --- Step 2: Preprocess for contours ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    washer_objects = []
    hexnut_objects = []
    screw_objects = []

    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue

        if cv2.matchShapes(c, ref_c, 1, 0.0) < 0.01:
            continue

        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box) if not imutils.is_cv2() else cv2.cv.BoxPoints(box)
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
            continue  # ignore tiny noise

        obj_centerX = (tl[0] + br[0]) / 2
        obj_centerY = (tl[1] + br[1]) / 2
        obj_center = (obj_centerX, obj_centerY)

        distance_pixels = dist.euclidean(ref_center, obj_center)
        distance_inches = distance_pixels / pixelsPerMetric

        if distance_inches < 0.5:
            continue

        object_area = dimA * dimB  # square inches

        if object_area < (0.5 * 0.5):
            hexnut_objects.append({
                "box": box,
                "dimA": dimA,
                "dimB": dimB
            })
        else:
            aspect_ratio = dimA / dimB if dimB != 0 else 0
            if aspect_ratio > 1.1 or aspect_ratio < 0.5:
                screw_objects.append({
                    "box": box,
                    "dimA": dimA,
                    "dimB": dimB
                })
            else:
                washer_objects.append({
                    "box": box,
                    "dimA": dimA,
                    "dimB": dimB
                })

    # --- Check for hex nut overlaps with washers ---
    valid_hexnuts = []

    for hex_obj in hexnut_objects:
        hx_min_x = int(min(hex_obj["box"][:,0]))
        hx_max_x = int(max(hex_obj["box"][:,0]))
        hx_min_y = int(min(hex_obj["box"][:,1]))
        hx_max_y = int(max(hex_obj["box"][:,1]))

        overlap = False

        for wash_obj in washer_objects:
            wx_min_x = int(min(wash_obj["box"][:,0]))
            wx_max_x = int(max(wash_obj["box"][:,0]))
            wx_min_y = int(min(wash_obj["box"][:,1]))
            wx_max_y = int(max(wash_obj["box"][:,1]))

            if (hx_min_x < wx_max_x and hx_max_x > wx_min_x and
                hx_min_y < wx_max_y and hx_max_y > wx_min_y):
                overlap = True
                break

        if not overlap:
            valid_hexnuts.append(hex_obj)

    # --- Draw and count ---

    if washer_objects:
        frames_with_washer += 1
        for obj in washer_objects:
            (tl, tr, br, bl) = obj["box"]
            dimA = obj["dimA"]
            dimB = obj["dimB"]

            cv2.drawContours(orig, [obj["box"].astype("int")], -1, (0, 255, 0), 2)
            cv2.putText(orig, "Washer", (int(tl[0]), int(tl[1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            cv2.putText(orig, "{:.1f}in".format(dimA),
                        (int(tl[0] - 15), int(tl[1] - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(orig, "{:.1f}in".format(dimB),
                        (int(tr[0] + 10), int(tr[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if valid_hexnuts:
        frames_with_hexnut += 1
        for obj in valid_hexnuts:
            (tl, tr, br, bl) = obj["box"]
            dimA = obj["dimA"]
            dimB = obj["dimB"]

            cv2.drawContours(orig, [obj["box"].astype("int")], -1, (255, 255, 0), 2)
            cv2.putText(orig, "Hex nut", (int(tl[0]), int(tl[1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

            cv2.putText(orig, "{:.1f}in".format(dimA),
                        (int(tl[0] - 15), int(tl[1] - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(orig, "{:.1f}in".format(dimB),
                        (int(tr[0] + 10), int(tr[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if screw_objects:
        frames_with_screw += 1
        for obj in screw_objects:
            (tl, tr, br, bl) = obj["box"]
            dimA = obj["dimA"]
            dimB = obj["dimB"]

            cv2.drawContours(orig, [obj["box"].astype("int")], -1, (0, 0, 255), 2)
            cv2.putText(orig, "Screw", (int(tl[0]), int(tl[1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            cv2.putText(orig, "{:.1f}in".format(dimA),
                        (int(tl[0] - 15), int(tl[1] - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(orig, "{:.1f}in".format(dimB),
                        (int(tr[0] + 10), int(tr[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- Show the result ---
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

cap.release()
cv2.destroyAllWindows()

# --- Final results ---
print("\n[RESULTS]")
if total_processed_frames > 0:
    print(f"Total frames processed: {total_processed_frames}")
    print(f"Frames with Washer: {frames_with_washer} ({100 * frames_with_washer / total_processed_frames:.2f}%)")
    print(f"Frames with Screw: {frames_with_screw} ({100 * frames_with_screw / total_processed_frames:.2f}%)")
    print(f"Frames with Hex nut (no overlap with washers): {frames_with_hexnut} ({100 * frames_with_hexnut / total_processed_frames:.2f}%)")
else:
    print("No frames were processed.")

