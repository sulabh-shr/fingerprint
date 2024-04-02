import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from fingerprint.utils import crop_fingerprint, equalize_clahe
    from fingerprint.utils import iterate_mmfv_files

    from fingerprint_enhancer import FingerprintImageEnhancer

    crop = True
    hist = True
    enhance = False
    out_ext = '.jpg'

    overwrite = False

    root = r"D:\workspace\datasets\MMFV-10th-start_0"
    # out_root = r"D:\workspace\datasets\MMFV-10th-start_0_processed"
    out_root = r"D:\workspace\datasets\MMFV-10th-start_0_cropped"

    for d in iterate_mmfv_files(root, out_root=out_root):
        in_path = d['img_path']
        out_dir = d['out_dir']
        frame, _ = os.path.splitext(d['frame'])
        out_path = os.path.join(out_dir, frame + out_ext)

        if not overwrite and os.path.isfile(out_path):
            print(f'File exists: {out_path}')
            continue
        print(out_path)
        img = cv2.imread(in_path)

        cropped_img = img
        if crop:
            try:
                cropped_img = crop_fingerprint(img, segment=True, channels='BGR', viz=False)
            except Exception as e:
                print(f'Error: {in_path}')
                print(str(e))
                continue

        enhanced_img = cropped_img
        if hist:
            enhanced_img = equalize_clahe(cropped_img, channels='BGR')

        img_out_uint8 = enhanced_img
        if enhance:
            img_gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)

            # Contrast stretching:
            # https://stackoverflow.com/questions/72004972/extract-ridges-and-valleys-from-finger-image
            k = 5
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (k, k))  # larger than the width of the widest ridges
            o = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, se)  # locally lowest grayvalue
            c = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, se)  # locally highest grayvalue
            gray = (img_gray - o) / (c - o + 1e-6)

            image_enhancer = FingerprintImageEnhancer()
            try:
                # Fingerprint Enhancement
                img_out = image_enhancer.enhance(gray)
            except (IndexError, ValueError) as e:
                print(f'Error for file: {in_path}')
                # fig, ax = plt.subplots(1, 3, figsize=(20, 8))
                # ax[0].imshow(img[:, :, ::-1])
                # ax[1].imshow(cropped_img[:, :, ::-1])
                # ax[2].imshow(gray, cmap='gray')
                # plt.show()
                continue
            img_out_uint8 = img_out.astype(np.uint8) * 255

        if img_out_uint8.shape[0] > 0 and img_out_uint8.shape[1] > 0:
            cv2.imwrite(out_path, img_out_uint8)
