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
    path = "./data/voc/Annotations/2007_000027.xml"
    with open(path, encoding="utf-8") as file:
        tree = ET.parse(file)
    root = tree.getroot()
    f = root.find("filename").text
    print(f)