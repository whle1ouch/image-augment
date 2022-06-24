from multiprocessing.sharedctypes import Value
from PIL import Image
import numpy as np


class Operation(object):
    
    def __init__(self, probability):
        self.probability = probability
    
    def perform(self, image, *args, **kwargs):
        raise ValueError("operation is not implemented.")
        
    def perform_opration(self, images, *args, **kwargs):
        performed_images = []
        for image in images:
            performed_images.append(self.perform(image, *args,**kwargs))
        return performed_images
    

class RandomNoise(Operation):
    
    def perform(self, image, var=None):
        if var is None:
            var =  0.1
        elif var < 0:
            raise ValueError("the var that used to generate random noise shoule be positive")
        out = image + np.random.randint(0, int(var ** 0.5), size=image.shape)
        out = out.astype(np.uint8)
        return out
    
class GaussianNoise(Operation):
    
    def perform(self, image):
        ...

class CutOut(Operation):
    
    def perform(self, image):
        ...
        
class DropOut(Operation):
    
    def perform(self, image):
        ...
        
class Salt(Operation):
    
    def perform(self, image):
        ...
        
class Cartton(Operation):
    
    def perform(self, image):
        ...
        
class Blend(Operation):
    
    def perform(self, image):
        ...
        
class GaussianBlur(Operation):
    
    def perform(self, image):
        ...
        
class MotionBlur(Operation):
    
    def perform(self, image):
        ...
        
class ColorTemp(Operation):
    
    def perform(self, image):
        ...
        
class HistogramEqualization(Operation):
    
    def perform(self, image):
        ...
        
class HorizontalFlip(Operation):
    
    def perform(self, image):
        ...
        
class VerticalFlip(Operation):
    
    def perform(self, image):
        ...
        
class Scale(Operation):
    
    def perform(self, image):
        ...
        
        
class RandomScale(Operation):
    
    def perform(self, image):
        ...
        
        
class RandomTranslation(Operation):
    
    def perform(self, image):
        ...
        
class HSVTransform(Operation):
    
    def perform(self, image):
        ...
        
class PerspectiveTransform(Operation):
    
    def perform(self, image):
        ...
        
        
class RandomContrast(Operation):
    
    def perform(self, image):
        ...
        
class EdgeEnhence(Operation):
    
    def perform(self, image):
        ...
        
class RandomBright(Operation):
    
    def perform(self, image):
        ...
        
class MaxPooling(Operation):
    
    def perform(self, image):
        ...
        
class AveragePooling(Operation):
    
    def perform(self, image):
        ...
        
class RandomCrop(Operation):
    
    def perform(self, image):
        ...
        
class RandomPad(Operation):
    
    def perform(self, image):
        ...