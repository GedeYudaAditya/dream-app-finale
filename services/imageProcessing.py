# for image processing
import base64

# Kebutuhan Untuk Prosesing Model
import cv2
import numpy as np

# Import constant.py untuk mendapatkan nilai imageWidth dan imageHeight
from config.constant import imageHeight, imageWidth, minDistance


def base64_to_image(base64_string):
    """
    The base64_to_image function accepts a base64 encoded string and returns an image.
    The function extracts the base64 binary data from the input string, decodes it, converts 
    the bytes to numpy array, and then decodes the numpy array as an image using OpenCV.

    :param base64_string: Pass the base64 encoded image string to the function
    :return: An image
    """
    base64_data = base64_string.split(",")[1]
    image_bytes = base64.b64decode(base64_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def find_if_close(cnt1, cnt2):
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < minDistance:
                return True
            elif i == row1-1 and j == row2-1:
                return False


def objectCentroid(contours):
    LENGTH = len(contours)
    status = np.zeros((LENGTH, 1))
    # print(LENGTH)

    for i, cnt1 in enumerate(contours):
        x = i
        if i != LENGTH-1:
            for j, cnt2 in enumerate(contours[i+1:]):
                x = x+1
                dist = find_if_close(cnt1, cnt2)
                if dist == True:
                    val = min(status[i], status[x])
                    status[x] = status[i] = val
                else:
                    if status[x] == status[i]:
                        status[x] = i+1

    allCont = []
    maximum = int(status.max())+1
    # print(f"ini maksimum: {maximum}")
    for i in range(maximum):
        pos = np.where(status == i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            allCont.append(hull)

    sorted_contours = sorted(allCont, key=cv2.contourArea, reverse=True)
    largest_item = sorted_contours[0]

    hull = cv2.convexHull(largest_item)

    # find centroid
    M = cv2.moments(hull)
    try:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    except ZeroDivisionError:
        cx = 0
        cy = 0

    return cx, cy, hull


def finalize(prediction, frame):
    imageOri = cv2.imread(prediction)
    imageOri = cv2.resize(imageOri, (imageWidth, imageHeight))
    image = cv2.cvtColor(imageOri, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(image, 127, 255, 0)

    # h, w, c = imageOri.shape
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)

    imageResult = imageOri

    # check if contour is available or not
    if len(contours) == 0:
        # print('kosong')

        # status kemuncul -> tidak muncul = False
        status = False
    else:

        # status kemuncul -> muncul = True
        status = True

        # centroid object
        clx, cly, largest_item = objectCentroid(contours)
        printCentroidObject = str(clx) + ', ' + str(cly)

        lx, ly, lw, lh = cv2.boundingRect(largest_item)
        cv2.rectangle(imageResult, (lx, ly), (lx+lw, ly+lh), (255, 255, 0), 2)

        # shape
        cv2.circle(imageResult, (clx, cly), radius=0,
                   color=(0, 255, 0), thickness=7)
        cv2.putText(imageResult, printCentroidObject, (clx, cly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # image Combine
    frame = cv2.imread(frame)
    frame = cv2.resize(frame, (imageWidth, imageHeight))
    hasilBlend = cv2.addWeighted(frame, 1, imageResult, 1, 0)
    cv2.imwrite("hasilBlend.png", hasilBlend)

    hasilBlend = cv2.imread('hasilBlend.png', cv2.IMREAD_UNCHANGED)
    cv2.imwrite("hasilBlend.png", hasilBlend)

    imageResult = hasilBlend

    return imageResult, status
