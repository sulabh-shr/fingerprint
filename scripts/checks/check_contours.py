import os
import cv2
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt


def rotate_contour(contour, m):
    x = contour[:, 0, 0]
    y = contour[:, 0, 1]
    x2 = m[0, 0] * x + m[0, 1] * y + m[0, 2]
    y2 = m[1, 0] * x + m[1, 1] * y + m[1, 2]
    return x2, y2


def crop_fingerprint(img, viz=False):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get largest contour
    ret, thresh = cv2.threshold(img_grey, 127, 255, 0)
    contours, hierarchy = cv2.findContours(~thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit ellipse on largest contour
    (x, y), (MA, ma), angle = cv2.fitEllipse(largest_contour)
    if angle > 90:
        angle = -(180 - angle)

    vertical_image, result, rot_contour_x, rot_contour_y = rotate_and_crop_with_contour(
        img, x, y, angle, largest_contour)

    # Fit rectangle if ellipse does not work
    if result.shape[0] == 0 or result.shape[1] == 0:
        print('Using rectangle for approximation')
        (x, y), (MA, ma), angle = cv2.minAreaRect(largest_contour)
        if angle > 90:
            angle = -(180 - angle)
        vertical_image, result, rot_contour_x, rot_contour_y = rotate_and_crop_with_contour(
            img, x, y, angle, largest_contour)

    if viz:
        o1 = img.copy()
        cv2.drawContours(o1, contours, -1, (0, 0, 255), 5)

        o2 = img.copy()
        cv2.drawContours(o2, [largest_contour], 0, (255, 0, 0), 10)

        o3 = vertical_image.copy()
        rot_largest_contour = np.zeros_like(largest_contour)
        rot_largest_contour[:, 0, 0] = rot_contour_x
        rot_largest_contour[:, 0, 1] = rot_contour_y
        cv2.drawContours(o3, [rot_largest_contour], 0, (255, 0, 0), 10)

        fig, ax = plt.subplots(1, 4, figsize=(20, 10))
        ax[0].imshow(o1[:, :, ::-1])
        ax[1].imshow(o2[:, :, ::-1])
        ax[2].imshow(o3[:, :, ::-1])
        # ax[3].imshow(result[:, :, ::-1])

        plt.show()
        plt.close()

    return result


def rotate_and_crop_with_contour(img, x, y, angle, contour):
    # Rotate the image about ellipse so that it is vertical
    rot_mat = cv2.getRotationMatrix2D((x, y), angle, 1.0)
    vertical_image = cv2.warpAffine(img, rot_mat, img.shape[1::-1],
                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Rotate contour separately but identical to the image
    rot_contour_x, rot_contour_y = rotate_contour(contour, rot_mat)

    # Find lowest point in rotated contour
    lowest_idx = np.argmax(rot_contour_y)
    lowest_x, lowest_y = rot_contour_x[lowest_idx], rot_contour_y[lowest_idx]
    # print(lowest_x, lowest_y)
    lowest_x, lowest_y = round(lowest_x), round(lowest_y)

    # Crop based on the lowest point and fixed offset
    dx = 300
    dy1 = 550
    dy2 = 50
    x1 = max(lowest_x - dx, 0)
    x2 = min(lowest_x + dx, img.shape[1])
    y1 = max(lowest_y - dy1, 0)
    y2 = min(lowest_y + dy2, img.shape[0])

    cropped_image = vertical_image[y1:y2, x1:x2, :]
    return vertical_image, cropped_image, rot_contour_x, rot_contour_y


def sharpen(img, viz=False):
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    alpha = 0.3
    sharpened = cv2.addWeighted(img, 1 + alpha, blurred, -alpha, 0)
    if viz:
        fig, ax = plt.subplots(1, 3, figsize=(15, 10))
        ax[0].imshow(img[:, :, ::-1])
        ax[1].imshow(blurred[:, :, ::-1])
        ax[2].imshow(sharpened[:, :, ::-1])
        plt.show()
    return sharpened


if __name__ == '__main__':
    root = r"D:\workspace\datasets\MMFV-Dataset\70\2\f1\Roll"
    root = r"D:/workspace/datasets/MMFV-25th\61\1\f1\Pitch"
    files = natsorted(os.listdir(root))

    for i in files:
        # path = os.path.join(root, '2_1.jpg')
        path = os.path.join(root, i)
        print(path)
        img = cv2.imread(path)
        out = crop_fingerprint(img, viz=True)

        # sharpen(out, viz=True)
