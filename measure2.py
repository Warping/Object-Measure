# Measurement and object classification code (updated to ignore objects near reference)
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# --- Arguments ---
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
ap.add_argument("-w", "--width", type=float, default=1.0,
    help="known width of the red reference object in inches (default 1 inch)")
args = vars(ap.parse_args())

# --- Load and resize image ---
image = cv2.imread(args["image"])

# --- Resize image if too large BEFORE anything else ---
max_width = 1000
height, width = image.shape[:2]

if width > max_width:
    scale = max_width / width
    image = cv2.resize(image, (int(width * scale), int(height * scale)))



# --- preprocess image ---
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# --- Find all external contours ---
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)


##This  a preview of contpours
preview = image.copy()  # Copy of the original image to draw on
cv2.drawContours(preview, cnts, -1, (0, 255, 0), 2)  # Draw all contours in green
max_width = 1000
height, width = preview.shape[:2]

if width > max_width:
    scale = max_width / width
    resized = cv2.resize(preview, (int(width * scale), int(height * scale)))
else:
    resized = preview.copy()

cv2.imshow("Contour Preview", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
##end




orig = image.copy()

# --- Step 1: Find red square as reference object ---
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
    print("[ERROR] No red reference object found!")
    exit()

ref_c = max(ref_cnts, key=cv2.contourArea)
ref_box = cv2.minAreaRect(ref_c)
ref_box = cv2.boxPoints(ref_box) if not imutils.is_cv2() else cv2.cv.BoxPoints(ref_box)
ref_box = np.array(ref_box, dtype="int")
ref_box = perspective.order_points(ref_box)

# Calculate pixels per metric
(tl, tr, br, bl) = ref_box
(tltrX, tltrY) = midpoint(tl, tr)
(blbrX, blbrY) = midpoint(bl, br)
(tlblX, tlblY) = midpoint(tl, bl)
(trbrX, trbrY) = midpoint(tr, br)

dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
pixelsPerMetric = dB / args["width"]

# --- Reference box center ---
ref_centerX = (tl[0] + br[0]) / 2
ref_centerY = (tl[1] + br[1]) / 2
ref_center = (ref_centerX, ref_centerY)

# --- Draw reference box and label ---
cv2.drawContours(orig, [ref_box.astype("int")], -1, (0, 0, 255), 2)
cv2.putText(orig, "Reference object", (int(tl[0]), int(tl[1]) - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
cv2.putText(orig, "{:.1f}in".format(args["width"]),
            (int(tr[0] + 10), int(tr[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# --- Step 2: Measure and classify other objects ---
objects = []

for i, c in enumerate(cnts):
    if cv2.contourArea(c) < 100:
        continue

    # Skip the reference contour itself
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

    # --- Filter out small objects ---
    if dimA < 0.5 and dimB < 0.5:
        continue

    # --- Calculate object's center ---
    obj_centerX = (tl[0] + br[0]) / 2
    obj_centerY = (tl[1] + br[1]) / 2
    obj_center = (obj_centerX, obj_centerY)

    # --- Distance to reference ---
    distance_pixels = dist.euclidean(ref_center, obj_center)
    distance_inches = distance_pixels / pixelsPerMetric

    # --- Skip if too close to reference (false object) ---
    if distance_inches < 0.5:
        continue

    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    objects.append({
        "box": box,
        "dimA": dimA,
        "dimB": dimB,
        "contour": c
    })

# --- Labeling ---
for obj in objects:
    box = obj["box"]
    dimA = obj["dimA"]
    dimB = obj["dimB"]
    c = obj["contour"]

    label = "Hex nut"  # Default

    aspect_ratio = dimA / dimB if dimB != 0 else 0

    if aspect_ratio > 1.1 or aspect_ratio < 0.5:
        label = "Screw"
    else:
        perimeter = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0

        is_circular = circularity > 0.7

        has_hole = False
        cnts2, hierarchy_data = cv2.findContours(edged.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy_data[0] if hierarchy_data is not None else []

        for h_idx, h in enumerate(hierarchy):
            if h_idx < len(cnts2) and np.array_equal(c, cnts2[h_idx]):
                if h[2] != -1:
                    has_hole = True
                break

        if is_circular and has_hole:
            label = "Washer"

    (tl, tr, br, bl) = box
    cv2.putText(orig, label, (int(tl[0]), int(tl[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    cv2.putText(orig, "{:.1f}in".format(dimA),
                (int(tl[0] - 15), int(tl[1] - 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(orig, "{:.1f}in".format(dimB),
                (int(tr[0] + 10), int(tr[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# --- Show final result ---
cv2.imshow("Labeled Objects", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
