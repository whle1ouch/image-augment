from zipfile import Path
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from src.operation import *
from src.images import *
from skimage import transform
import glob, os
from xml.etree import ElementTree as ET
from src.images import ImageProcesser, ImageSource


if __name__ == "__main__":
    path = "./data/voc/"
    p = ImageProcesser("./data/output/", target_type="pose",  save_to_disk=True)
    p.scan_voc(path)
    p.process()
    p1 = ImageProcesser(target_type="pose")
    p1.load("./data/output/")
    print(p1)
    print(p1.image_sources[0])
    