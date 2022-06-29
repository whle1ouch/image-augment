from zipfile import Path
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
    path = "./data/voc/"
    p = VOCDataset(path, label_type="segement_class")
    print(p)
    for image in p.image_sources:
        print(image)
    