from ssl import HAS_SNI
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageColor
import numpy as np
import cv2


class Operation(object):
    
    def __init__(self, p):
        self.probability = p
    
    def perform(self, image):
        raise ValueError("operation is not implemented.")
        
    def perform_opration(self, images):
        performed_images = []
        for image in images:
            performed_images.append(self.perform(image))
        return performed_images
    

class RandomNoise(Operation):
    
    def __init__(self, p, max_value=None):
        super().__init__(p)
        if max_value is None:
            max_value = 20
        elif max_value < 0:
            raise ValueError("the max value that used to generate random noise shoule be positive.")
        self.max_value = int(max_value)
    
    def perform(self, image):
        arr = np.array(image)
        noise = np.random.randint(0, self.var, size=arr.shape)
        out_arr = image + noise
        out = Image.fromarray(out_arr.astype(np.uint8), mode=image.mode)
        return out


class GaussianNoise(Operation):
    
    def __init__(self, p, mean=None, std=None):
        super().__init__(p)
        if mean is None:
            mean = 0
        if std is None:
            std = 5
        self.mean = mean
        self.std = std
    
    def perform(self, image):
        arr = np.array(image)
        noise = np.random.normal(self.mean, self.std, size=arr.shape)
        out_arr = arr + noise
        out = Image.fromarray(out_arr.astype(np.uint8), mode=image.mode)
        return out


class CutOut(Operation):
    
    def __init__(self, p, num=None, rect_ratio=None):
        super().__init__(p)
        if num is None:
            num = 4
        if not isinstance(num, int) or num < 1:
            raise ValueError("cut rect numbers should be positive int.")
        if rect_ratio is None:
            rect_ratio = 0.1
        if rect_ratio <= 0 or rect_ratio >= 1:
            raise ValueError("cutout rect ratio shold be in (0, 1).")
        self.num = num
        self.rect_ratio = rect_ratio
    
    def perform(self, image, num=4, rect_ratio=0.1):
        arr = np.array(image)
        shape = arr.shape
        h, w = shape[0], shape[1]
        rh, rw = max(int(h * rect_ratio / 2), 1), max(int(w * rect_ratio / 2), 1)
        out_arr = arr.copy()
        if shape == 2:
            color_size = (1,)
        else:
            color_size = (3,)
        for _ in range(num):
            x = np.random.randint(rh, h-rh+1)
            y = np.random.randint(rw, w-rw+1)
            color = np.random.randint(0, 50, size=color_size)
            left, right =  x-rh, x+rh+1
            upper, bottom = y-rw, y+rw+1
            out_arr[left:right, upper:bottom, ...] = color
        out = Image.fromarray(out_arr, mode=image.mode)
        return out
            
        
class DropOut(Operation):
    
    def __init__(self, p, rate=None):
        super().__init__(p)
        if rate is None:
            rate = 0.1
        if rate <= 0 or rate >=1:
            raise ValueError("dropout rate should be a float in (0, 1).")
        self.rate = rate
        
    
    def perform(self, image):
        arr = np.array(image)
        out_arr = arr.copy()
        dropout = np.random.random(size=arr.shape) < self.rate
        out_arr[dropout] = 0
        out = Image.fromarray(out_arr, mode=image.mode)
        return out
        
        
class SaltPepperNoise(Operation):
    
    def __init__(self, p, amount=None, salt_vs_pepper=None):
        super().__init__(p)
        if amount is None:
            amount = 0.05
        if salt_vs_pepper is None:
            salt_vs_pepper = 0.5
        if amount <= 0 or amount >=1:
            raise ValueError("amount of  should be in (0, 1).")
        if salt_vs_pepper < 0 or salt_vs_pepper > 1:
            raise ValueError("salt rate should be positive in [0, 1].")
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper
    
    def perform(self, image):
        arr = np.array(image)
        out_arr = arr.copy()
        flipped = np.random.choice([True, False], size=arr.shape[:2], p=[self.amount, 1-self.amount])
        salted = np.random.choice([True, False], size=arr.shape[:2], p=[self.salt_vs_pepper, 1-self.salt_vs_pepper])
        peppered = ~salted
        out_arr[flipped & salted, ...] = 255
        out_arr[flipped & peppered, ...] = 0
        out = Image.fromarray(out_arr, mode=image.mode)
        return out


class Cartoon(Operation):
    
    def __init__(self, p, sigma_s=None, sigma_r=None):
        super().__init__(p)
        if sigma_s is None:
            sigma_s = 100
        if sigma_r is None:
            sigma_r = 0.1
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r
    
    def perform(self, image):
        arr = np.array(image)
        out_arr = cv2.stylization(arr, sigma_s=self.sigma_s, sigma_r=self.sigma_r)
        out = Image.fromarray(out_arr, mode=image.mode)
        return out 
        

class Blend(Operation):
    
    def __init__(self, p, direct="right", mask_weight=None):
        super().__init__(p)
        if direct not in ("left", "right", "upper", "bottom"):
            raise ValueError("""mask direct should be in one of ('left', 'right', 'upper', 'bottom'),
                             which means the gradient color in the direct from black to white.""")
        if mask_weight is None:
            mask_weight = 0.4
        if mask_weight <= 0 or mask_weight >= 1:
            raise ValueError("mask_weight should be in (0, 1), whose default is 0.4")
        self.direct = direct
        self.mask_weight = mask_weight
        
    
    def perform(self, image):
        w, h = image.size
        arr = np.array(image)
        if self.direct in ("left", "right"):
            _t = np.linspace(0, 255, w).reshape((1, w))
            if self.direct == "left":
                _t = _t[:, ::-1]
            mask = np.repeat(_t, h, 0)
        else:
            _t = np.linspace(0, 255, h).reshape((h, 1))
            if self.direct == "upper":
                _t = _t[::-1, :]
            mask = np.repeat(_t, w, 1)
        if len(arr.shape) > 2:
            mask = mask[:, :, np.newaxis]
            mask = np.repeat(mask, arr.shape[2], 2)
        out_arr = arr * (1 - self.mask_weight) + mask * (self.mask_weight)
        out = Image.fromarray(out_arr.astype(np.uint8), mode=image.mode)
        return out


class GaussianBlur(Operation):
    
    def __init__(self, p, ksize=None, sigma_X=None):
        super().__init__(p)
        if ksize is None:
            ksize = (5, 5)
        elif isinstance(ksize, int):
            if ksize < 1:
                raise ValueError("kernel size should be positive.")
            ksize = (ksize, ksize)
        if sigma_X is None:
            sigma_X = 5
        self.ksize = ksize
        self.sigma_X = sigma_X
        

    def perform(self, image):
        arr = np.array(image)
        out_arr = cv2.GaussianBlur(arr, self.ksize, self.sigma_X)
        out = Image.fromarray(out_arr, mode=image.mode)
        return out


class MotionBlur(Operation):
    
    def __init__(self, p, degree=None, angle=None):
        super().__init__(p)
        if degree is None:
            degree = 10
        if angle is None:
            angle = 20
        self.degree = degree 
        self.angle = angle
    
    
    def perform(self, image):
        arr = np.array(image)
        M = cv2.getRotationMatrix2D((self.degree / 2, self.degree / 2), self.angle, 1)
        motion_blur_kernel = np.diag(np.ones(self.degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (self.degree, self.degree))
        motion_blur_kernel = motion_blur_kernel / self.degree
        out_arr = cv2.filter2D(arr, -1, motion_blur_kernel)
        out = Image.fromarray(out_arr, mode=image.mode)
        return out



class RandomColorTemp(Operation):
    #TODO: is this right?
    def perform(self, image):
        # color temp: lambda r,g,b: r-b/2 - g -256
        arr = np.array(image)
        out_arr = arr.copy()
        ratio = np.random.uniform(0.8, 1.2)
        # out[:,:,0] = out[:,:,0] + level
        # out[:,:,1] = out[:,:,1] - level 
        out_arr[:,:,2] = out_arr[:,:,2] * ratio
        out = Image.fromarray(out_arr.astype(np.uint8), mode=image.mode)
        return out 
        
                
class HistogramEqualization(Operation):
    
    def perform(self, image):
        out = ImageOps.equalize(image)
        return out
        
class HorizontalFlip(Operation):
    
    def perform(self, image):
        out = ImageOps.mirror(image)
        return out
        
class VerticalFlip(Operation):
    
    def perform(self, image):
        out = ImageOps.flip(image)
        return out
        
class Scale(Operation):
    
    def __init__(self, p, scale_factor=None, keep_shape=True):
        super().__init__(p)
        if scale_factor is None:
            scale_factor = np.random.uniform(0.6, 1.0)
        if scale_factor <= 0 or scale_factor >= 1:
            raise ValueError("resize scale ratio should be in (0, 1) in order to keep she shape")
        self.scale_factor = scale_factor
        self.keep_shape = keep_shape
        
    
    def perform(self, image):
        w, h = image.size 
        new_w, new_h =  int(w * self.scale_factor), int(h * self.scale_factor)
        resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        if self.keep_shape:
            gray_color = ImageColor.getcolor('gray', image.mode)
            out = Image.new(image.mode, image.size, gray_color)
            out.paste(resized, ((w - new_w) // 2, (h - new_h)//2))
        else:
            out = resized
        return out

        
class RandomScale(Operation):
    
    def __init__(self, p, keep_shape=True):
        super().__init__(p)
        self.keep_shape = keep_shape
    
    def perform(self, image):
        w, h = image.size
        scale_w, scale_h = np.random.uniform(0.5, 1.0), np.random.uniform(0.5, 1.0)
        new_w, new_h = int(w * scale_w), int(h * scale_h)
        resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        if self.keep_shape:
            gray_color = ImageColor.getcolor('gray', image.mode)
            out = Image.new(image.mode, image.size, gray_color)
            out.paste(resized, ((w - new_w) // 2, (h - new_h)//2))
        else:
            out = resized
        return out
        
        
class RandomTranslation(Operation):
    
    def perform(self, image):
        w, h = image.size
        trans_w = np.random.randint(-w // 4 + 1, w // 4)
        trans_h = np.random.randint(-h // 4 + 1, h // 4)
        # positive: out[trans_w: w] = image[0: (w-trans_w)]
        # negetive: out[0 :(w+trans_w)] = image[-trans_w: w]
        arr = np.array(image)
        out_arr = np.ones_like(arr, dtype=np.uint8) * 80
        out_arr[max(0, trans_h): min(h, h+trans_h),max(0, trans_w): min(w, w+trans_w), ...] = arr[
                max(0, -trans_h): min(h-trans_h, h), max(0, -trans_w): min(w-trans_w, w),...]
        out = Image.fromarray(out_arr, mode=image.mode)
        return out
        
        
class RandomColor(Operation):
    
    def perform(self, image):
        factor = np.random.uniform(0.5, 1.5)
        enhencer = ImageEnhance.Color(image)
        out = enhencer.enhance(factor)
        return out
    

class RandomHSV(Operation):
    
    def perform(self, image):
        hsv =  image.convert("HSV")
        arr = np.array(hsv).astype(np.float32)
        hf = np.random.uniform(0.9, 1.1)
        sf = np.random.uniform(0.8, 1.2)
        vf = np.random.uniform(0.8, 1.2)
        arr[..., 0] *= hf
        arr[..., 1] *= sf
        arr[..., 2] *= vf
        arr = arr.astype(np.uint8)
        out_hsv = Image.fromarray(arr, mode="HSV")
        out = out_hsv.convert(image.mode)
        return out 


    
class Sharpe(Operation):
    
    def perform(self, image):
        enhencer = ImageEnhance.Sharpness(image)
        out = enhencer.enhance(2)
        return out
        
                
class PerspectiveTransform(Operation):
    
    
    def __init__(self, p, skew_type="random", magnitude=None):
        super().__init__(p)
        skew_type = skew_type.lower()
        if skew_type not in ("random", "tilt", "tilt_left_right", "tilt_top_bottom", "corner"):
            raise ValueError('skew type should be in one of ("random", "tilt", "tilt_left_right", "tilt_top_bottom", "corner")')
        if skew_type == "random":
            skew_type = np.random.choice(["all", "tilt", "tilt_left_right", "tilt_top_bottom", "corner"])
        if magnitude is None:
            magnitude = 0.5
        self.skew_type = skew_type
        self.magnitude = magnitude
        
    
    def perform(self, image):
        w, h = image.size
        x1 = 0
        x2 = h
        y1 = 0
        y2 = w
        original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]
        max_skew_amount = max(w, h)
        max_skew_amount = int(np.ceil(max_skew_amount * self.magnitude))
        skew_amount = np.random.randint(1, max_skew_amount+1)
        
        if self.skew_type in ("tilt", "tilt_left_right", "tilt_top_bottom"):
            if self.skew_type == "tilt":
                skew_direction = np.random.randint(0, 4)
            elif self.skew_type == "tilt_left_right":
                skew_direction = np.random.randint(0, 2)
            elif self.skew_type == "tilt_top_bottom":
                skew_direction = np.random.randint(2, 4)
            if skew_direction == 0:
                # Left Tilt
                new_plane = [(y1, x1 - skew_amount),  # Top Left
                            (y2, x1),                # Top Right
                            (y2, x2),                # Bottom Right
                            (y1, x2 + skew_amount)]  # Bottom Left
            elif skew_direction == 1:
                # Right Tilt
                new_plane = [(y1, x1),                # Top Left
                            (y2, x1 - skew_amount),  # Top Right
                            (y2, x2 + skew_amount),  # Bottom Right
                            (y1, x2)]                # Bottom Left
            elif skew_direction == 2:
                # Forward Tilt
                new_plane = [(y1 - skew_amount, x1),  # Top Left
                            (y2 + skew_amount, x1),  # Top Right
                            (y2, x2),                # Bottom Right
                            (y1, x2)]                # Bottom Left
            elif skew_direction == 3:
                # Backward Tilt
                new_plane = [(y1, x1),                # Top Left
                            (y2, x1),                # Top Right
                            (y2 + skew_amount, x2),  # Bottom Right
                            (y1 - skew_amount, x2)]  # Bottom Left
        elif self.skew_type == "corner":
            skew_direction = np.random.randint(0, 8)
            if skew_direction == 0:
                # Skew possibility 0
                new_plane = [(y1 - skew_amount, x1), (y2, x1), (y2, x2), (y1, x2)]
            elif skew_direction == 1:
                # Skew possibility 1
                new_plane = [(y1, x1 - skew_amount), (y2, x1), (y2, x2), (y1, x2)]
            elif skew_direction == 2:
                # Skew possibility 2
                new_plane = [(y1, x1), (y2 + skew_amount, x1), (y2, x2), (y1, x2)]
            elif skew_direction == 3:
                # Skew possibility 3
                new_plane = [(y1, x1), (y2, x1 - skew_amount), (y2, x2), (y1, x2)]
            elif skew_direction == 4:
                # Skew possibility 4
                new_plane = [(y1, x1), (y2, x1), (y2 + skew_amount, x2), (y1, x2)]
            elif skew_direction == 5:
                # Skew possibility 5
                new_plane = [(y1, x1), (y2, x1), (y2, x2 + skew_amount), (y1, x2)]
            elif skew_direction == 6:
                # Skew possibility 6
                new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1 - skew_amount, x2)]
            elif skew_direction == 7:
                # Skew possibility 7
                new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2 + skew_amount)]
        elif self.skew_type == "all":
            corners = dict()
            corners["top_left"] = (y1 - np.random.randint(1, skew_amount+1), x1 - np.random.randint(1, skew_amount+1))
            corners["top_right"] = (y2 + np.random.randint(1, skew_amount+1), x1 - np.random.randint(1, skew_amount+1))
            corners["bottom_right"] = (y2 + np.random.randint(1, skew_amount+1), x2 + np.random.randint(1, skew_amount+1))
            corners["bottom_left"] = (y1 - np.random.randint(1, skew_amount+1), x2 + np.random.randint(1, skew_amount+1))
            new_plane = [corners["top_left"], corners["top_right"], corners["bottom_right"], corners["bottom_left"]]
        matrix = []
        for p1, p2 in zip(new_plane, original_plane):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
        A = np.matrix(matrix, dtype=np.float)
        B = np.array(original_plane).reshape(8)
        perspective_skew_coefficients_matrix = np.dot(np.linalg.pinv(A), B)
        perspective_skew_coefficients_matrix = np.array(perspective_skew_coefficients_matrix).reshape(8)
        out = image.transform(image.size,
                              Image.PERSPECTIVE,
                              perspective_skew_coefficients_matrix,
                              resample=Image.BICUBIC)
        return out
        
        
class RandomContrast(Operation):
    
    def perform(self, image):
        factor = np.random.uniform(0.5, 1.5)
        enhencer = ImageEnhance.Contrast(image)
        out = enhencer.enhance(factor)
        return out
        
        
class EdgeEnhence(Operation):
    
    def perform(self, image):
        out = image.filter(ImageFilter.EDGE_ENHANCE)
        return out
        
class RandomBright(Operation):
    
    def perform(self, image):
        factor = np.random.uniform(0.7, 1.3)
        enhencer = ImageEnhance.Brightness(image)
        out = enhencer.enhance(factor)
        return out
        
class MaxPooling(Operation):
    
    def perform(self, image, ksize=5):
        out = image.filter(ImageFilter.MaxFilter(ksize))
        return out
        
        
class MedianPooling(Operation):
    
    def perform(self, image, ksize=5):
        out = image.filter(ImageFilter.MedianFilter(ksize))
        return out
        
        
class RandomCrop(Operation):
    
    def __init__(self, p, centre=True):
        super().__init__(p)
        self.centre = centre
    
    def perform(self, image):
        ratio = np.random.uniform(0.5, 0.9)
        w, h = image.size
        w_new, h_new = int(w * ratio), int(h * ratio)
        left_shift = np.random.randint(0, w-w_new)
        down_shift = np.random.randint(0, h-h_new)
        if self.centre:
            out = image.crop(((w/2)-(w_new/2), (h/2)-(h_new/2), (w/2)+(w_new/2), (h/2)+(h_new/2)))
        else:
            out = image.crop((left_shift, down_shift, w_new + left_shift, h_new + down_shift))
        return out
        
    
        
class RandomPad(Operation):
    
    def __init__(self, p, centre=True):
        super().__init__(p)
        self.centre = centre
    
    def perform(self, image):
        cut_ratio = np.random.uniform(0.05, 0.2)
        w, h = image.size
        w_cut, h_cut = int(w * cut_ratio), int(h * cut_ratio)
        arr = np.array(image)
        out_arr = np.ones_like(arr, dtype=np.uint8) * 80
        if self.centre:
            w_shift, h_shift = w_cut // 2, h_cut // 2
            print(h_cut)
            out_arr[h_shift : h - h_shift, w_shift : w - w_shift, ...] = arr[h_shift : h - h_shift, w_shift : w - w_shift, ...]
        else:
            out_arr[: h - h_cut, : w - w_cut,  ...] = arr[: h - h_cut, : w - w_cut, ...]
        out = Image.fromarray(out_arr, mode=image.mode)
        return out 