from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from src.operation import *
from skimage import transform

if __name__ == "__main__":
    image_pil = Image.open("./data/2.jpg")
    op = RandomPad(0.4)
    imagec_pil = op.perform(image_pil)
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_pil)
    plt.subplot(1, 2, 2)
    plt.imshow(imagec_pil)
    plt.show()
