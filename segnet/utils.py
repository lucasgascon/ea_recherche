import matplotlib.pyplot as plt 
from PIL import Image as im

import os
import random

def infer_dir(model, dir_path, n):
    list_img = os.listdir(dir_path)[:n]
    random.shuffle(list_img)
    for img in list_img:
        img_path = dir_path + img
        print(img_path)
        out = model.predict_segmentation(
            inp= img_path,
            # out_fname="out.png"
        )

        plt.imshow(im.open(img_path))
        plt.show()
        plt.imshow(out)
        plt.show()

def infer_img(model, img_path):
    out = model.predict_segmentation(
        inp= img_path,
        out_fname="out.png"
    )

    plt.imshow(im.open(img_path))
    plt.show()
    plt.imshow(out)
    plt.show()