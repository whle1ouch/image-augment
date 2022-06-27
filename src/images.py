import os, glob
from PIL import Image
from xml.etree import ElementTree as ET



class ImageData:
    
    
    def __init__(self):
        self.path = None
        self.pil = None
        self.resolution = None
        self.format = None
        self._label = None
        
    def fromarray(self, arr, mode="RGB"):
        self.pil = Image.fromarray(arr, mode=mode)
        self.resolution = self.pil.size
        
    def fromfile(self, path):
        format = path.split(".")[-1]
        if format not in ['jpg', 'bmp', 'jpeg', 'gif', 'img', 'png', 'tiff', 'tif']:
            raise ValueError(f"image format of {format} is not supported.")
        self.pil = Image.open(path)
        self.resolution = self.pil.size
        self.format = format
    
    def fromcv2(self, mat):
        self.fromarray(mat, "RGB")
        
    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self, val):
        self._label = val


class ImageSet:
    
    def __init__(self):
        ...
    
    def read_from_voc(self, voc_folder):
        vf = os.path.abspath(voc_folder)
        
        
        
        
    def read_from_folder(self):
        ...