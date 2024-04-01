import cv2
import numpy as np
import matplotlib.pyplot as plt


def rotate_contour(contour, m):
    x = contour[:, 0, 0]
    y = contour[:, 0, 1]
    x2 = m[0, 0] * x + m[0, 1] * y + m[0, 2]
    y2 = m[1, 0] * x + m[1, 1] * y + m[1, 2]
    return x2, y2


def crop_fingerprint(img: np.ndarray, segment: bool, channels='RGB', verbose=False, viz=False):
    if channels == 'BGR':  # cv2.imread
        # color_mode = cv2.COLOR_BGR2GRAY
        color_mode = cv2.COLOR_BGR2HSV
    elif channels == 'RGB':  # Image.open
        # color_mode = cv2.COLOR_RGB2GRAY
        color_mode = cv2.COLOR_RGB2HSV
    else:
        raise ValueError(f'Invalid channel type: {channels}')

    # Get largest contour
    # img_gray = cv2.cvtColor(img, color_mode)
    # ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    # contours, hierarchy = cv2.findContours(~thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_hsv = cv2.cvtColor(img, color_mode)
    img_gray = img_hsv[:, :, 1]
    ret, thresh = cv2.threshold(img_gray, 50, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit ellipse on largest contour
    (x, y), (MA, ma), angle = cv2.fitEllipse(largest_contour)
    if angle > 90:
        angle = -(180 - angle)

    # Make none contour pixels zero
    img_org = img
    if segment:
        if viz:
            img_org = img.copy()
        contour_mask = np.zeros_like(img_gray)
        cv2.drawContours(contour_mask, [largest_contour], 0, color=(255, 255, 255), thickness=cv2.FILLED)
        img[contour_mask == 0, :] = (0, 0, 0)

    vertical_img, cropped_img, rot_contour_x, rot_contour_y = rotate_and_crop_with_contour(
        img, x, y, angle, largest_contour)

    # Fit rectangle if ellipse does not work
    if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
        if verbose:
            print('Using rectangle for contour rotation estimation')
        (x, y), (MA, ma), angle = cv2.minAreaRect(largest_contour)
        if angle > 90:
            angle = -(180 - angle)
        vertical_img, cropped_img, rot_contour_x, rot_contour_y = rotate_and_crop_with_contour(
            img, x, y, angle, largest_contour)

    if viz:
        o1 = img_org.copy()
        # cv2.drawContours(o1, contours, -1, (0, 0, 255), 5)

        o2 = img_org.copy()
        cv2.drawContours(o2, [largest_contour], 0, (255, 0, 0), 10)

        o3 = vertical_img.copy()
        rot_largest_contour = np.zeros_like(largest_contour)
        rot_largest_contour[:, 0, 0] = rot_contour_x
        rot_largest_contour[:, 0, 1] = rot_contour_y
        cv2.drawContours(o3, [rot_largest_contour], 0, (255, 0, 0), 10)

        fig, ax = plt.subplots(1, 4, figsize=(20, 10))
        # Image.open is used to load image so channel inversion not required
        ax[0].imshow(o1)
        ax[0].set_title(f'Original Image')
        ax[1].imshow(o2)
        ax[1].set_title(f'Largest Contour')
        ax[2].imshow(o3)
        ax[2].set_title(f'Vertically Aligned')
        ax[3].imshow(cropped_img)
        ax[3].set_title(f'Cropped')

        plt.show()
        plt.close()

    return cropped_img


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
