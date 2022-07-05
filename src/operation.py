from re import A
from ssl import HAS_SNI
from tkinter import W
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageColor
import numpy as np
import cv2


class Operation(object):
    
    def __init__(self, p):
        self.probability = p
        
    def perform(self, image):
        raise ValueError("operation is not implemented.")
    
    def perform_batch(self, images):
        new_images = list()
        for image in images:
            new_images.append(self.perform(image))
        return new_images
    
    def perform_with_box(self, image, boxes):
        raise ValueError("""this operation may change the coordinate of each pixel in the image, 
                          and this function is not implemented.""")
         
    def perform_with_segment(self, image, segment):
        raise ValueError("""this operation may change the coordinate of each pixel in the image, 
                          and this function is not implemented.""")
         
    
class PixelOperation(Operation):
    
    def perform_with_box(self, image, boxes):
        new_image = self.perform(image)
        return new_image, boxes
    
    def perform_with_segment(self, image, segment):
        out = self.perform(image)
        return image, segment
        

class MorphOperation(Operation):
    
    ...
    
    

class RandomNoise(PixelOperation):
    
    def __init__(self, p, max_value=None):
        super().__init__(p)
        if max_value is None:
            max_value = 10
        elif max_value < 0:
            raise ValueError("the max value that used to generate random noise shoule be positive.")
        self.max_value = int(max_value)
    
    def perform(self, image):
        arr = np.array(image)
        noise = np.random.randint(0, self.max_value, size=arr.shape)
        out_arr = image + noise
        out = Image.fromarray(out_arr.astype(np.uint8), mode=image.mode)
        return out



class GaussianNoise(PixelOperation):
    
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


class CutOut(PixelOperation):
    
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
            
        
class DropOut(PixelOperation):
    
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
        
        
class SaltPepperNoise(PixelOperation):
    
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


class Cartoon(PixelOperation):
    
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
        

class Blend(PixelOperation):
    
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


class GaussianBlur(PixelOperation):
    
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


class MotionBlur(PixelOperation):
    
    def __init__(self, p, degree=None, angle=None):
        super().__init__(p)
        if degree is None:
            degree = 10
        if angle is None:
            angle = 20
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        self.motion_blur_kernel = motion_blur_kernel / degree
    
    
    def perform(self, image):
        arr = np.array(image)
        out_arr = cv2.filter2D(arr, -1, self.motion_blur_kernel)
        out = Image.fromarray(out_arr, mode=image.mode)
        return out



class RandomColorTemp(PixelOperation):
    
    def __init__(self, p):
        super().__init__(p)
        self.ratio = 1.
        
    #TODO: is this right?
    def perform(self, image):
        self.ratio = np.random.uniform(0.8, 1.2)
        # color temp: lambda r,g,b: r-b/2 - g -256
        arr = np.array(image)
        out_arr = arr.copy()
        # out[:,:,0] = out[:,:,0] + level
        # out[:,:,1] = out[:,:,1] - level 
        out_arr[:,:,2] = out_arr[:,:,2] * self.ratio
        out = Image.fromarray(out_arr.astype(np.uint8), mode=image.mode)
        return out 
        
                
class HistogramEqualization(PixelOperation):
    
    def perform(self, image):
        out = ImageOps.equalize(image)
        return out
        
class HorizontalFlip(MorphOperation):
    
    def perform(self, image):
        out = ImageOps.mirror(image)
        return out
    
    def perform_with_box(self, image, boxes):
        out = self.perform(image)
        w, _ = image.size
        flip_box = []
        for box in boxes:
            if len(box) == 4:
                xmin, ymin, xmax, ymax = box
                flip_box.append((w - xmax, ymin, w-xmin, ymax))
            else:
                flip_box.append(())
        return out, flip_box
    
    def perform_with_segment(self, image, segment):
        out, out_segment = self.perform_batch([image, segment])
        return out, out_segment
    
    
    
        
class VerticalFlip(MorphOperation):
    
    def perform(self, image):
        out = ImageOps.flip(image)
        return out
    
    def perform_with_box(self, image, boxes):
        out = self.perform(image)
        _, h = image.size
        flip_box = []
        for box in boxes:
            if len(box) == 4:
                xmin, ymin, xmax, ymax = box
                flip_box.append((xmin, h - ymax, xmax, h - ymin))
            else:
                flip_box.append(())
        return out, flip_box

    def perform_with_segment(self, image, segment):
        out, out_segment = self.perform_batch([image, segment])
        return out, out_segment
        
        

    
        
        
class Scale(MorphOperation):
    
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
        resized = image.resize((new_w, new_h), Image.BICUBIC)
        if self.keep_shape:
            gray_color = ImageColor.getcolor('gray', image.mode)
            out = Image.new(image.mode, image.size, gray_color)
            out.paste(resized, ((w - new_w) // 2, (h - new_h)//2))
        else:
            out = resized
        return out
    
    def perform_with_box(self, image, boxes):
        out = self.perform(image)
        w, h = image.size
        scale_box = []
        if self.keep_shape:
            shift_x , shift_y = int((1-self.scale_factor) * w) // 2, int((1-self.scale_factor) * h) // 2
        else:
            shift_x, shift_y = 0, 0
        for box in boxes:
            if len(box) == 4:
                xmin, ymin, xmax, ymax = map(lambda x: int(x * self.scale_factor), box)
                scale_box.append((xmin + shift_x, ymin+shift_y, xmax+shift_x, ymax+shift_y))
            else:
                scale_box.append(())
        return out, scale_box
    
    def perform_with_segment(self, image, segment):
        out, out_segment = self.perform_batch([image, segment])
        return out, out_segment

class FixScale(MorphOperation):
    
    def __init__(self, p, size):
        super().__init__(p)
        self.size = size
        
        
    def perform(self, image):
        w, h = image.size
        iw, ih = self.size
        scale = min(iw / w, ih / h)
        new_w, new_h = round(scale * w), round(scale * h)
        resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        out = Image.new(mode=image.mode, size=self.fixed_size)
        out.paste(resized, ((iw - new_w) // 2, (ih - new_h) // 2))
        return out
    
    def perform_with_box(self, image, boxes):
        w, h = image.size
        iw, ih = self.size
        scale_factor = min(iw / w, ih / h)
        new_w, new_h = round(scale_factor * w), round(scale_factor * h)
        resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        out = Image.new(mode=image.mode, size=self.fixed_size)
        shift_x, shift_y = (iw - new_w) // 2, (ih - new_h) // 2
        out.paste(resized, (shift_x, shift_y))
        scale_boxes = []
        for box in boxes:
            if len(box) == 4:
                xmin, ymin, xmax, ymax = map(lambda x: int(x * scale_factor), box)
                scale_boxes.append((xmin + shift_x, ymin+shift_y, xmax+shift_x, ymax+shift_y))
            else:
                scale_boxes.append(())
                
        return out, scale_boxes
    
    
    def perform_with_segment(self, image, segment):
        out, out_segment = self.perform_batch([image, segment])
        return out, out_segment
        

        
class RandomScale(MorphOperation):
    
    def __init__(self, p, keep_shape=True):
        super().__init__(p)
        self.keep_shape = keep_shape
        self.new_w = 0
        self.new_h = 0
        
    def random_sample(self, image):
        w, h = image.size
        scale_w = np.random.uniform(0.5, 1.0) 
        scale_h = np.random.uniform(0.5, 1.0)
        self.new_w, self.new_h = int(w * scale_w), int(h * scale_h)
        
    def do(self, image):
        if self.new_w == 0 or self.new_h == 0:
            return image
        w, h = image.size
        resized = image.resize((self.new_w, self.new_h), Image.BICUBIC)
        if self.keep_shape:
            gray_color = ImageColor.getcolor('gray', image.mode)
            out = Image.new(image.mode, image.size, gray_color)
            out.paste(resized, ((w - self.new_w) // 2, (h - self.new_h) // 2))
        else:
            out = resized
        return out
        
    
    def perform(self, image):
        self.random_sample(image)
        return self.do(image)

    
    def perform_with_box(self, image, boxes):
        out = self.perform(image)
        w, h = image.size
        scale_box = []
        if self.keep_shape:
            shift_x , shift_y = int((1-self.scale_w) * w) // 2, int((1-self.scale_h) * h) // 2
        else:
            shift_x, shift_y = 0, 0
        for box in boxes:
            if len(box) == 4:
                xmin, ymin, xmax, ymax = box
                xmin, xmax = int(xmin * self.scale_w), int(xmax * self.scale_w) 
                ymin, ymax = int(ymin * self.scale_h), int(ymax * self.scale_h)
                scale_box.append((xmin + shift_x, ymin+shift_y, xmax+shift_x, ymax+shift_y))
            else:
                scale_box.append(())
        return out, scale_box
    
    def perform_with_segment(self, image, segment):
        self.random_sample(image)
        return self.do(image), self.do(segment)
    
    
        
        
class RandomTranslation(MorphOperation):
    
    def __init__(self, p):
        super().__init__(p)
        self.trans_w = 0
        self.trans_h = 0
        
    def random_sample(self, image):
        w, h = image.size
        self.trans_w = np.random.randint(-w // 4 + 1, w // 4)
        self.trans_h = np.random.randint(-h // 4 + 1, h // 4)
        
    def do(self, image):
        w, h = image.size
        arr = np.array(image)
        out_arr = np.ones_like(arr, dtype=np.uint8) * 80
        out_arr[max(0, self.trans_h): min(h, h + self.trans_h),max(0, self.trans_w): min(w, w+ self.trans_w), ...] = arr[
                max(0, -self.trans_h): min(h-self.trans_h, h), max(0, -self.trans_w): min(w-self.trans_w, w),...]
        out = Image.fromarray(out_arr, mode=image.mode)
        return out 
        
     
    def perform(self, image):
        self.random_sample(image)
        return self.do(image)
        
    def perform_with_box(self, image, boxes):
        out = self.perform(image)
        w, h = image.size
        trans_boxes = []
        for box in boxes:
            if len(box) == 4:
                xmin, ymin, xmax, ymax = box
                xmin, xmax = max(xmin+self.trans_w, 0), min(xmax+self.trans_w, w-1)
                ymin, ymax = max(ymin+self.trans_h, 0), min(ymax+self.trans_h, h-1)
                if xmin < xmax and ymin < ymax:
                    trans_boxes.append((xmin, ymin, xmax, ymax))
                else:
                    trans_boxes.append(())
            else:
                trans_boxes.append(())
        return out, trans_boxes 
    
    def perform_with_segment(self, image, segment):
        self.random_sample(image)
        return self.do(image), self.do(segment)
    
class RandomColor(PixelOperation):
    
    def __init__(self, p):
        super().__init__(p)
        self.factor = 1
    
    def perform(self, image):
        self.factor = np.random.uniform(0.5, 1.5)
        enhencer = ImageEnhance.Color(image)
        out = enhencer.enhance(self.factor)
        return out
    

class RandomHSV(PixelOperation):
        
    
    def perform(self, image):
        hsv =  image.convert("HSV")
        hf = np.random.uniform(0.9, 1.1)
        sf = np.random.uniform(0.8, 1.2)
        vf = np.random.uniform(0.8, 1.2)
        arr = np.array(hsv).astype(np.float32)
        arr[..., 0] *= hf
        arr[..., 1] *= sf
        arr[..., 2] *= vf
        arr = arr.astype(np.uint8)
        out_hsv = Image.fromarray(arr, mode="HSV")
        out = out_hsv.convert(image.mode)
        return out 


    
class Sharpe(PixelOperation):
    
    def perform(self, image):
        enhencer = ImageEnhance.Sharpness(image)
        out = enhencer.enhance(2)
        return out
        
                
class PerspectiveTransform(MorphOperation):
    
    
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
        self._matrix = None
        
    def random_sample(self, image):
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
            else:
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
        original_plane = np.array(original_plane, dtype=np.float32)
        new_plane = np.array(new_plane, dtype=np.float32)
        self._matrix = cv2.getPerspectiveTransform(original_plane, new_plane)
        
    def do(self, image):
        if self._matrix is None:
            return image
        arr = np.array(image)
        img_warp = cv2.warpPerspective(arr, self._matrix, image.size)
        out = Image.fromarray(img_warp, mode=image.mode)
        return out
        
        
        
    
    def perform(self, image):
        self.random_sample(image)
        return self.do(image)
        
    
    def perform_with_box(self, image, boxes):
        out = self.perform(image)
        perspective_box = list()
        w, h = image.size
        for box in boxes:
            if len(box) == 4:
                left, top, right, bottom = box
                points = np.array([[left, top], [left, bottom], [right, top], [right, bottom]], dtype=np.float32)
                new_points = cv2.perspectiveTransform(points.reshape((-1, 1, 2)), self._matrix)
                new_points = new_points.reshape((4, 2))
                print(len(new_points), len(new_points[0]))
                left = min(new_points[0, 0], new_points[1, 0])
                top = min(new_points[0, 1], new_points[2, 1])
                right = max(new_points[2, 0], new_points[3, 0])
                bottom = max(new_points[1, 1], new_points[3, 1])
                left, right = max(0, left), min(right, w-1)
                top, bottom = max(0, top), min(bottom, h-1)
                if left >= right or top >= bottom:
                    perspective_box.append(())
                else:
                    perspective_box.append((left, top, right, bottom))
            else:
                perspective_box.append(())
        return out, perspective_box
    
    def perform_with_segment(self, image, segment):
        self.random_sample(image)
        return self.do(image), self.do(segment)
             
               
        
class RandomContrast(PixelOperation):
    
    def perform(self, image):
        factor = np.random.uniform(0.5, 1.5)
        enhencer = ImageEnhance.Contrast(image)
        out = enhencer.enhance(factor)
        return out
        
        
class EdgeEnhence(PixelOperation):
    
    def perform(self, image):
        out = image.filter(ImageFilter.EDGE_ENHANCE)
        return out
        
class RandomBright(PixelOperation):    
    
    def perform(self, image):
        factor = np.random.uniform(0.7, 1.3)
        enhencer = ImageEnhance.Brightness(image)
        out = enhencer.enhance(factor)
        return out
        
class MaxPooling(PixelOperation):
    
    def __init__(self, p, ksize=5):
        super().__init__(p)
        self.ksize = ksize
    
    def perform(self, image):
        out = image.filter(ImageFilter.MaxFilter(self.ksize))
        return out
        
        
class MedianPooling(PixelOperation):
    
    def __init__(self, p, ksize=5):
        super().__init__(p)
        self.ksize = ksize
    
    def perform(self, image):
        out = image.filter(ImageFilter.MedianFilter(self.ksize))
        return out
        
        
class Crop(Operation):
    
    def __init__(self, p, box, shift=False):
        super().__init__(p)
        if len(box) != 4:
            raise ValueError("""crop box should be a list or tuple with 4 elements such as (left, top, right, bottom), 
                             which means the cropping percentage of each direction""")
        self._crop_left, self._crop_top, self._crop_right, self._crop_bottom = box
        if shift:
            self._shift_w = np.random.uniform(-self._crop_left, self._crop_right)
            self._shift_h = np.random.uniform(-self._crop_top, self._crop_bottom)
        else:
            self._shift_w = 0
            self._shift_h = 0
        
    def perform(self, image):
        w, h = image.size
        left_crop, right_crop = int(w * self._crop_left), int(w * self._crop_right)
        top_crop, bottom_crop = int(h * self._crop_top), int(h * self._crop_bottom)
        croped = image.crop((left_crop, top_crop, w - right_crop, h - bottom_crop))
        w_shift, h_shift = int(w * self._shift_w), int(h * self._shift_h)
        gray_color = ImageColor.getcolor('gray', image.mode)
        out = Image.new(mode=image.mode, size=(w, h), color=gray_color)
        out.paste(croped, (left_crop + w_shift, top_crop + h_shift))
        return out
    
    def perform_with_box(self, image, boxes):
        w, h = image.size
        left_crop, right_crop = int(w * self._crop_left), int(w * self._crop_right)
        top_crop, bottom_crop = int(h * self._crop_top), int(h * self._crop_bottom)
        croped = image.crop((left_crop, top_crop, w - right_crop, h - bottom_crop))
        w_shift, h_shift = int(w * self._shift_w), int(h * self._shift_h)
        gray_color = ImageColor.getcolor('gray', image.mode)
        out = Image.new(mode=image.mode, size=(w, h), color=gray_color)
        out.paste(croped, (left_crop + w_shift, top_crop + h_shift))
        crop_boxes = []
        for box in boxes:
            if len(box) == 4:
                left, top, right, bottom = box
                left, right = max(left, left_crop), min(right, w - right_crop)
                top, bottom = max(top, top_crop), min(bottom, h - bottom_crop)
                if left >= right or top >= bottom:
                    continue
                left, right = left + w_shift, right + w_shift
                top, bottom = top + h_shift, bottom + h_shift
                crop_boxes.append((left, top, right, bottom))
            else:
                crop_boxes.append(())
        return out, crop_boxes
    
    def perform_with_segment(self, image, segment):
        out, out_segment = self.perform_batch([image, segment])
        return out, out_segment
        
        

class RandomCrop(Operation):
    
    def __init__(self, p, percentage=0.4, centre=False, shift=False):
        super().__init__(p)
        self.percentage = percentage
        self.centre = centre
        self.shift = shift
        self._crop_left = 0
        self._crop_right = 0
        self._crop_top = 0
        self._crop_bottom = 0
        self._shift_w = 0
        self._shift_h = 0
    
    def random_sample(self, image):
        w, h = image.size
        if self.centre:
            crop_percentages = np.random.uniform(0, self.percentage, size=(2, ))
            self._crop_left = self._crop_right = int(crop_percentages[0] * w) // 2
            self._crop_top =  self._crop_bottom = int(crop_percentages[0] * h) // 2
        else:
            crop_percentages = np.random.uniform(0, self.percentage / 2, size=(4, ))
            self._crop_left = int(crop_percentages[0] * w )
            self._crop_top = int(crop_percentages[1] * h) 
            self._crop_right = int(crop_percentages[2] * w)
            self._crop_bottom = int(crop_percentages[3] * h)
        if self.shift:
            self._shift_w = np.random.randint(-self._crop_left, self._crop_right)
            self._shift_h = np.random.randint(-self._crop_top, self._crop_bottom)
        else:
            self._shift_w = 0
            self._shift_h = 0
            
    def do(self, image):
        w, h = image.size
        croped = image.crop((self._crop_left, self._crop_top, w - self._crop_right, h - self._crop_bottom))
        gray_color = ImageColor.getcolor('gray', image.mode)
        out = Image.new(mode=image.mode, size=(w, h), color=gray_color)
        out.paste(croped, (self._crop_left + self._shift_w, self._crop_top + self._shift_h))
        return out

    
    def perform(self, image):
        self.random_sample(image)
        return self.do(image)
    
    def perform_with_box(self, image, boxes):
        self.random_sample(image)
        out = self.perform(image)
        w, h = image.size
        crop_boxes = []
        for box in boxes:
            if len(box) == 4:
                left, top, right, bottom = box
                left, right = max(left, self._crop_left), min(right, w - self._crop_right)
                top, bottom = max(top, self._crop_top), min(bottom, h - self._crop_bottom)
                if left >= right or top >= bottom:
                    continue
                left, right = left + self._shift_w, right + self._shift_w
                top, bottom = top + self._shift_h, bottom + self._shift_h
                crop_boxes.append((left, top, right, bottom))
            else:
                crop_boxes.append(())
        return out, crop_boxes
    
    def perform_with_segment(self, image, segment):
        self.random_sample()
        return self.do(image), self.do(segment)
        
        

        
            
        
        
    
        
