from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from src.operation import RandomNoise

if __name__ == "__main__":
    image_pil = Image.open("./data/2007_000027.jpg")
    image = np.array(image_pil, dtype=np.uint8)
    op = RandomNoise(0.4)
    imagec = op.perform(image, var=20)
    imagec_pil = Image.fromarray(imagec)
    plt.subplot(1, 2, 1)
    plt.imshow(image_pil)
    plt.subplot(1, 2, 2)
    plt.imshow(imagec_pil)
    plt.show()