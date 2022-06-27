from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from src.operation import *
from src.images import *
from skimage import transform
import glob, os

if __name__ == "__main__":
    image = Image.open("./data/1.jpg")
    