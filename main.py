from zipfile import Path
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from src.operation import *
from skimage import transform
import glob, os
from xml.etree import ElementTree as ET
from src.dataset import ObjectTarget
from src.utilies import draw_box


if __name__ == "__main__":
    # path = "./data/voc/"
    # image = Image.open("./data/1.jpg")
    # box = [(133, 88, 197, 123), (104, 78, 375, 183), \
    #     (195, 180, 213, 229), (26, 189, 44, 238)]
    # op = RandomCrop(0.4)
    # new_image, new_box = op.perform_with_box(image, box)
    # image = draw_box(image, box)
    # new_image = draw_box(new_image, new_box, 4)
    # plt.subplot(1, 2, 1)
    # plt.imshow(image)
    # plt.subplot(1, 2, 2)
    # plt.imshow(new_image)
    # plt.show()
    a = ObjectTarget("1", (1, 2, 3, 4))
    b = ObjectTarget("2", (2, 3, 4, 5))
    c = ObjectTarget("3", (5, 6, 7, 8))
    b.c = c
    a.c = b
    print(a.all_boxes())