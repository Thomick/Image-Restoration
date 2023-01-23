import cv2
import os

img_dir = "denoisedflickrSR/"
img_list = os.listdir(img_dir)

for img_path in img_list:
    print(img_path)
    img = cv2.imread(img_dir + img_path, cv2.IMREAD_UNCHANGED)
    width = int(img.shape[1] / 3.5)
    height = int(img.shape[0] / 3.5)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(img_dir + img_path[:-4] + "DS.png", img)
