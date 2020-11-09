import cv2 as cv
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import imutils

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def draw_feature_contours(img, pixelsPerMetric = 1.1, display_size_text = False):
    #img_inverted = False
    if(len(img.shape) > 2):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img
        print("converting the img to gray before finding contours ...")
    h, w = gray.shape
    #print("gray img shape: ", h , w)
    dilated = cv.dilate(gray, None, iterations = 1)
    eroded = cv.erode(dilated, None, iterations = 1)
    # if background larger that invert the img
    print("zero pixels: ",len(np.where(eroded == 0.0)))
    #count = h*w // 2
    #if (len(np.where(np.reshape(eroded,-1)==0)[0]) < count):
    #    eroded  = cv.bitwise_not(eroded)
    #    img_inverted = True
    cnts = cv.findContours(eroded.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #print("cnts : ", cnts)
    if len(cnts[0]) > 0:
        cnts = imutils.grab_contours(cnts)
        #print(cnts)
        (cnts, _) = contours.sort_contours(cnts)
        #print("sorted: ",cnts)
        for c in cnts:
            # compute the rotated bounding box of the contour
            orig = img
            # if the contour is not sufficiently large, ignore it
            if cv.contourArea(c) < 15:
                continue
            
            box = cv.minAreaRect(c)
            box = cv.cv.BoxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
            box = np.array(box, dtype="int")

            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            cv.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 1)

            # loop over the original points and draw them
            #for (x, y) in box:
            #    cv.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)

            # compute the midpoint between the top-left and bottom-left points,
            # followed by the midpoint between the top-righ and bottom-right
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            # compute the size of the object
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric

            # draw the object sizes on the image
            #(int(tltrX), int(tltrY+5))
            #(int(trbrX), int(trbrY-5))
            if display_size_text:
                cv.putText(orig, "{:.1f}mm".format(dimA),
                    (int(tlblX), int(tlblY)), cv.FONT_HERSHEY_SIMPLEX,
                    0.22, (255, 255, 255), 1)
                cv.putText(orig, "{:.1f}in".format(dimB),
                    (int(tltrX), int(tltrY)), cv.FONT_HERSHEY_SIMPLEX,
                    0.22, (255, 255, 255), 1)
            print("finished drawig contours to the image, returning ...")
        return orig
    else:
        return None
