import matplotlib.pyplot as plt 
from PIL import Image as im
import cv2

import os 
import random

def infer_dir(model, dir_path, n, k = None):
    list_img = os.listdir(dir_path)[:n]
    random.shuffle(list_img)
    for img in list_img:
        img_path = dir_path + img
        print(img_path)
        out = model.predict_segmentation(
            inp= img_path,
            # out_fname="out.png"
        )


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2),
                                        # sharex=True, sharey=True,
                                        )
        ax1.imshow(cv2.imread(img_path))
        ax1.axis('off')
        ax1.set_title('Image')
        ax2.imshow(out, cmap='gray')
        ax2.axis('off')
        ax2.set_title('Prédiction')
        fig.suptitle('Exemple de prédiction de PSPNet ré-entraîné sur ' + str(k) + ' images.')
        fig.tight_layout()
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