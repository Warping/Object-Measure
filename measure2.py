# Measurement and object classification code
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
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

# --- Load and preprocess image ---
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# --- Find contours ---
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
(cnts, _) = contours.sort_contours(cnts)

pixelsPerMetric = None
orig = image.copy()

# --- Measure objects and store data ---
objects = []

for i, c in enumerate(cnts):
	if cv2.contourArea(c) < 100:
		continue

	box = cv2.minAreaRect(c)
	box = cv2.boxPoints(box) if not imutils.is_cv2() else cv2.cv.BoxPoints(box)
	box = np.array(box, dtype="int")
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)

	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	if pixelsPerMetric is None:
		pixelsPerMetric = dB / args["width"]

	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric

	objects.append({
		"box": box,
		"dimA": dimA,
		"dimB": dimB,
		"contour": c
	})

# --- Washer detection: Find nested contours ---
cnts2, hierarchy_data = cv2.findContours(edged.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy_data[0] if hierarchy_data is not None else []

# --- Labeling ---
screw_height_threshold = 1.5  # Inches

for idx, obj in enumerate(objects):

	box = obj["box"]
	dimA = obj["dimA"]
	dimB = obj["dimB"]
	c = obj["contour"]

	label = "Hex nut"  # Default

	# 1️⃣ First object is the reference object
	if idx == 0:
		label = "Reference object"

	# 2️⃣ Screw: if height >= threshold
	elif dimA >= screw_height_threshold:
		label = "Screw"

	else:
		# --- Washer check ---

		# Circularity test
		perimeter = cv2.arcLength(c, True)
		area = cv2.contourArea(c)
		if perimeter == 0:
			circularity = 0
		else:
			circularity = 4 * np.pi * (area / (perimeter * perimeter))

		is_circular = circularity > 0.7

		# Hole check: does it have a child contour?
		has_hole = False

		for h_idx, h in enumerate(hierarchy):
			# h[2] = first child index
			# h[3] = parent index
			if h_idx < len(cnts2) and np.array_equal(c, cnts2[h_idx]):
				if h[2] != -1:
					has_hole = True
				break

		if is_circular and has_hole:
			label = "Washer"

	# --- Draw the label ---
	(tl, tr, br, bl) = box
	cv2.putText(orig, label, (int(tl[0]), int(tl[1]) - 15),
				cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

	# --- Draw dimensions ---
	cv2.putText(orig, "{:.1f}in".format(dimA),
				(int(tl[0] - 15), int(tl[1] - 30)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
	cv2.putText(orig, "{:.1f}in".format(dimB),
				(int(tr[0] + 10), int(tr[1])),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# --- Show the final labeled image ---
cv2.imshow("Labeled Objects", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
