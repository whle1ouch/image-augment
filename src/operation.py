from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np
import cv2


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
            var = 0.1 * 255
        elif var < 0:
            raise ValueError("the std that used to generate random noise shoule be positive")
        noise = np.random.randint(0, int(var), size=image.shape)
        out = image + noise
        out = out.astype(np.uint8)
        return out


class GaussianNoise(Operation):
    
    def perform(self, image, mean=None, std=None):
        if mean is None:
            mean = 0.
        if std is None:
            std = 0.1 * 255
        noise = np.random.normal(mean, std, size=image.shape)
        out = image + noise
        out = out.astype(np.uint8)
        return out


class CutOut(Operation):
    
    def perform(self, image, num=4, rect_ratio=0.1):
        if rect_ratio <= 0 or rect_ratio >= 1:
            raise ValueError("cutout rect ratio shold be in (0, 1)")
        shape = image.shape
        w, h = shape[0], shape[1]
        rw, rh = max(int(w * rect_ratio / 2), 1), max(int(h * rect_ratio / 2), 1)
        out = image.copy()
        if shape == 2:
            color_size = (1,)
        else:
            color_size = (3,)
        for _ in range(num):
            x = np.random.randint(rw, w-rw+1)
            y = np.random.randint(rh, h-rh+1)
            color = np.random.randint(0, 40, size=color_size)
            left, right =  x-rw, x+rw+1
            upper, bottom = y-rh, y+rh+1
            out[left:right, upper:bottom, ...] = color
        out = out.astype(np.uint8)
        return out
            
        
class DropOut(Operation):
    
    def perform(self, image, rate=0.1):
        if rate <= 0 or rate >=1:
            raise ValueError("dropout rate should be a float in (0, 1).")
        out = image.copy()
        dropout = np.random.random(size=image.shape) < rate
        out[dropout] = 0
        return out
        
        
class SaltPepperNoise(Operation):
    
    def perform(self, image, amount=0.1, salt_vs_pepper=1.):
        if amount <= 0 or amount >=1:
            raise ValueError("amount of  should be in (0, 1).")
        if salt_vs_pepper <= 0 or salt_vs_pepper > 1:
            raise ValueError("salt_vs_pepper should be positive.")
        out = image.copy()
        flipped = np.random.choice([True, False], size=image.shape[:2], p=[amount, 1-amount])
        salted = np.random.choice([True, False], size=image.shape[:2], p=[salt_vs_pepper, 1-salt_vs_pepper])
        peppered = ~salted
        out[flipped & salted, ...] = 255
        out[flipped & peppered, ...] = 0
        return out


class Cartoon(Operation):
    
    def perform(self, image):
        out = cv2.stylization(image, sigma_s=100, sigma_r=0.25)
        return out 
        

class Blend(Operation):
    
    def perform(self, image, direct="right", mask_weight=0.4):
        if direct not in ("left", "right", "upper", "bottom"):
            raise ValueError("""mask direct should be in one of ('left', 'right', 'upper', 'bottom'),
                             which means the gradient color in the direct from black to white.""")
        if mask_weight <= 0 or mask_weight >= 1:
            raise ValueError("mask_weight should be in (0, 1), whose default is 0.4")
        h, w = image.shape[:2]
        if direct in ("left", "right"):
            _t = np.linspace(0, 255, w).reshape((1, w))
            if direct == "left":
                _t = _t[:, ::-1]
            mask = np.repeat(_t, h, 0)
        else:
            _t = np.linspace(0, 255, h).reshape((h, 1))
            if direct == "upper":
                _t = _t[::-1, :]
            mask = np.repeat(_t, w, 1)
        if len(image.shape) > 2:
            mask = mask[:,:, np.newaxis]
            mask = np.repeat(mask, image.shape[2], 2)
        out = image * (1 - mask_weight) + mask * (mask_weight)
        out = out.astype(np.uint8)
        return out


class GaussianBlur(Operation):
    
    def perform(self, image, ksize=(5, 5), sigma_X=5):
        out = cv2.GaussianBlur(image, ksize, sigma_X)
        return out


class MotionBlur(Operation):
    
    def perform(self, image, degree=10, angle=20):
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = motion_blur_kernel / degree
        out = cv2.filter2D(image, -1, motion_blur_kernel)
        out = cv2.normalize(out, 0, 255, cv2.NORM_MINMAX)
        out = out.astype(np.uint8)
        return out



class ColorTemp(Operation):
    
    def perform(self, image):
        # color temp: lambda r,g,b: r-b/2 - g -256
        out = image.copy()
        ratio = np.random.choice([0.85, 1.15])
        #TODO: is this right?
        # out[:,:,0] = out[:,:,0] + level
        # out[:,:,1] = out[:,:,1] - level 
        out[:,:,2] = out[:,:,2] * ratio
        out = out.astype(np.uint8)
        return out 
        
                
class HistogramEqualization(Operation):
    
    def perform(self, image):
        if len(image.shape) == 2:
            out = cv2.equalizeHist(image)
        else:
            out = image.copy()
            for i in range(image.shape[2]):
                out[:, :, i] = cv2.equalizeHist(image[:, :, i])
        return out
        
class HorizontalFlip(Operation):
    
    def perform(self, image):
        out = np.fliplr(image)
        return out
        
class VerticalFlip(Operation):
    
    def perform(self, image):
        out = np.flipud(image)
        return out
        
class Scale(Operation):
    
    def perform(self, image, scale_factor=None, keep_shape=True):
        if scale_factor is None:
            scale_factor = (np.random.random() + 1) / 2
        if scale_factor <= 0 or scale_factor >= 1:
            raise ValueError("resize scale ratio should be in (0, 1) in order to keep she shape")
        w, h = image.shape[:2]
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        start_w, start_h = (w - new_w) // 2, (h - new_h) // 2
        resized = cv2.resize(image, (new_h, new_w), )
        if keep_shape:
            out = np.ones_like(image) * 100
            out[start_w:start_w+new_w, start_h : start_h+new_h, ...] = resized
        else:
            out = resized
        out = out.astype(np.uint8)
        return out

        
class RandomScale(Operation):
    
    def perform(self, image, keep_shape=True):
        scale_w, scale_h = (np.random.random() + 3) / 4, (np.random.random() + 3) / 4
        w, h = image.shape[:2]
        new_w, new_h = int(w * scale_w), int(h * scale_h)
        start_w, start_h = (w - new_w) // 2, (h - new_h) // 2
        resized = cv2.resize(image, (new_h, new_w))
        if keep_shape:
            out = np.ones_like(image) * 100
            out[start_w:start_w+new_w, start_h : start_h+new_h, ...] = resized
        else:
            out = resized
        out = out.astype(np.uint8)
        return out
        
        
class RandomTranslation(Operation):
    
    def perform(self, image):
        w, h = image.shape[:2]
        trans_w = np.random.randint(-w // 4 + 1, w // 4)
        trans_h = np.random.randint(-h // 4 + 1, h // 4)
        # positive: out[trans_w: w] = image[0: (w-trans_w)]
        # negetive: out[0 :(w+trans_w)] = image[-trans_w: w]
        out = np.ones_like(image) * 100
        out[max(0, trans_w): min(w, w+trans_w), 
            max(0, trans_h): min(h, h+trans_h), 
            ...] = image[
                max(0, -trans_w): min(w-trans_w, w),
                max(0, -trans_h): min(h-trans_h, h),
                ...
        ]
        out = out.astype(np.uint8)
        return out
        
        
class RandomHue(Operation):
    
    def perform(self, image):
        factor = 0.7 + np.random.random() / 2
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:,:, 0] = hsv[:,:, 0] * factor
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return out
        
                
class PerspectiveTransform(Operation):
    
    def perform(self, image, skew_type="random", magnitude=0.4):
        if skew_type not in ("random", "tilt", "tilt_left_right", "tilt_top_bottom", "corner"):
            raise ValueError('skew type should be in one of ("random", "tilt", "tilt_left_right", "tilt_top_bottom", "corner")')
        w, h = image.size
        x1 = 0
        x2 = h
        y1 = 0
        y2 = w

        original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]
        max_skew_amount = max(w, h)
        max_skew_amount = int(np.ceil(max_skew_amount * magnitude))
        skew_amount = np.random.randint(1, max_skew_amount+1)
        

        if skew_type == "random":
            skew_type = np.random.choice(["all", "tilt", "tilt_left_right", "tilt_top_bottom", "corner"])
        
        if skew_type in ("tilt", "tilt_left_right", "tilt_top_bottom"):
            if skew_type == "tilt":
                skew_direction = np.random.randint(0, 4)
            elif skew_type == "tilt_left_right":
                skew_direction = np.random.randint(0, 2)
            elif skew_type == "tilt_top_bottom":
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
        elif skew_type == "corner":
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
        elif skew_type == "all":
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
    
    def perform(self, image, factor=None):
        factor = np.random.uniform(0.7, 1.3)
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
        
        
class AveragePooling(Operation):
    
    def perform(self, image, ksize=5):
        out = image.filter(ImageFilter.MedianFilter(ksize))
        return out
        
        
class RandomCrop(Operation):
    
    def perform(self, image, centre=False):
        ratio = np.random.uniform(0.5, 0.9)
        w, h = image.size
        w_new, h_new = int(w * ratio), int(h * ratio)
        left_shift = np.random.randint(0, w-w_new)
        down_shift = np.random.randint(0, h-h_new)
        if centre:
            out = image.crop(((w/2)-(w_new/2), (h/2)-(h_new/2), (w/2)+(w_new/2), (h/2)+(h_new/2)))
        else:
            out = image.crop((left_shift, down_shift, w_new + left_shift, h_new + down_shift))
        return out
        
    
        
class RandomPad(Operation):
    
    def perform(self, image, centre=False):
        # cut_ratio = np.random.uniform(0.1, 0.2)
        cut_ratio = 0

        w, h = image.size
        w_cut, h_cut = int(w * cut_ratio), int(h * cut_ratio)
        print(w, h)
        arr = np.array(image)
        out_arr = np.ones_like(arr, dtype=np.uint8) * 80
        print(arr.shape)
        if centre:
            w_shift, h_shift = w_cut // 2, h_cut // 2
            print(h_cut)
            out_arr[h_shift : h - h_shift, w_shift : w - w_shift, ...] = arr[h_shift : h - h_shift, w_shift : w - w_shift, ...]
        else:
            out_arr[: w - w_cut, : h - h_cut, ...] = arr[: w - w_cut, : h - h_cut, ...]
        out_arr = np.tran
        out = Image.fromarray(out_arr)
        return out 