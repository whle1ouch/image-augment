from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from src.operation import *
from src.images import *
from skimage import transform
import glob, os
from xml.etree import ElementTree as ET
from src.images import ImageProcesser


if __name__ == "__main__":
    path = "./data/class"
    p = ImageProcesser(path, n_job=2, save_to_disk=True)
    print(p.image_sources)
    print(p.labels)
    b = p.process()
    print(b)