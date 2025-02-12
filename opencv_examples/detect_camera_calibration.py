# adapted from:
# https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration

import numpy as np
import time
import cv2
import glob
import sys
import traceback

PATH_TO_IMAGES = 'images/'
PATH_TO_CALIB = PATH_TO_IMAGES + 'calib/'

def detect_calibrate_and_undistort(images, image_to_fix, show_calibration=False):
    # termiantion criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 6))

        # If found, add object points, image points (after refining them)
        if ret is True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

            # Draw and display the corners
            if corners2 is not None:
                imgpoints.append(corners2)
                cv2.drawChessboardCorners(img, (7,6), corners2, ret)
            else:
                imgpoints.append(corners)
                cv2.drawChessboardCorners(img, (7,6), corners, ret)

            i = 1
            if show_calibration:
                try:
                    cv2.imshow('img', img)
                    if i == 12:
                        cv2.imwrite(PATH_TO_CALIB + 'calibinter.png',img)
                    time.sleep(.5)
                    cv2.destroyWindow('img')
                except Exception, e:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
                    print "error: ", e
                i += 1


    # ------------------------------------
    # Calibrate camera based on test images
    # -------------------------------------
    print "calibrating..."
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


    # ----------------------------
    # Undistort the image
    # ----------------------------
    print "undistorting..."
    img = cv2.imread(PATH_TO_IMAGES + image_to_fix)
    h, w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistorting method 1
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite(PATH_TO_CALIB + 'calibresult.png',dst)

    # undistorting method 2
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite(PATH_TO_CALIB + 'calibresult2.png', dst)

    if show_calibration:
        # show original
        cv2.imshow('img', img)

        # show undistortion method 1 result
        calibimg = cv2.imread(PATH_TO_CALIB + 'calibresult.png')
        cv2.imshow('calibrated', calibimg)
        cv2.moveWindow('calibrated', 500, 100)

        # show undistortion method 2 result
        calibimg2 = cv2.imread(PATH_TO_CALIB + 'calibresult2.png')
        cv2.imshow('calibrated2', calibimg2)
        cv2.moveWindow('calibrated2', 1100, 100)
        cv2.waitKey()

    # ----------------------------
    # Calculate reprojection error
    # ----------------------------
    mean_error = 0.0
    tot_error = 0.0
    for i in xrange(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = float(cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)) / float(len(imgpoints2))
        tot_error += error

    print "total error: ", float(mean_error) / float(len(objpoints))

if __name__ == "__main__":
    print "Usage:", sys.argv[0], "show calibration: True/False"
    print sys.argv
    show_calibration = sys.argv[1] == 'True'
    images = glob.glob(PATH_TO_IMAGES + 'left*.jpg')

    from timeit import Timer
    t = Timer(lambda: detect_calibrate_and_undistort(images, 'left12.jpg', show_calibration))
    repeats = 1
    print round(t.timeit(number=repeats) / repeats, 4), "seconds per run"
