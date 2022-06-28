from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from src.operation import *
from src.images import *
from skimage import transform
import glob, os
from xml.etree import ElementTree as ET
from src.images import NameDataset


if __name__ == "__main__":
    path = "./data/class"
    a = NameDataset(path)
    print(a.images)
    print(a.labels)