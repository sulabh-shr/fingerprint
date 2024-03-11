import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def viz_minutiae(img, minutiae, core=None):
    num_minutiae = minutiae.shape[0]
    minutiae.sort_values(by='score', inplace=True, ascending=False, ignore_index=True)
    print(minutiae)
    print(minutiae.loc[:, 'x'])
    dx, dy = 20, 20
    for i in range(5):
        x = round(minutiae.loc[i, 'x'])
        y = round(minutiae.loc[i, 'y'])
        score = minutiae.loc[i, 'score']
        color = np.random.randint(25, 200, (3,)).tolist()
        cv2.rectangle(img, (x - dx, y - dy), (x + dx, y + dy), color=color, thickness=3)
        cv2.circle(img, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
        print(x, y, score)

    if core is not None:
        cv2.rectangle(img, (core.loc[0, 'x1'], core.loc[0, 'y1']), (core.loc[0, 'x2'], core.loc[0, 'y2']),
                      color=(0, 0, 0), thickness=5)
    plt.figure(figsize=(15, 10))
    plt.imshow(img[:, :, ::-1])
    plt.show()


if __name__ == '__main__':

    minutiae_path = r"D:\workspace\mine\fingerflow\mmfv-25th-cropped\7\1\f1\Yaw\1_13-BGR-minutiae.csv"
    minutiae = pd.read_csv(minutiae_path)

    core_path = minutiae_path.replace('-BGR-minutiae', '-core')
    core = pd.read_csv(core_path)

    img_path = r"D:\workspace\mine\fingerflow\mmfv-25th-cropped\7\1\f1\Yaw\1_13.jpg"
    img = cv2.imread(img_path)

    viz_minutiae(img, minutiae, core)
