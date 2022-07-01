from PIL import Image, ImageDraw
import matplotlib.pyplot as plt



def resize(image, size):
    w, h = image.size
    iw, ih = size
    scale = min(iw / w, ih / h)
    new_w, new_h = round(scale * w), round(scale * h)
    resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    new_image = Image.new(mode=image.mode, size=size)
    new_image.paste(resized, ((iw - new_w) // 2, (ih - new_h) // 2))
    return new_image

def draw_box(image, bounding_boxs, thickness=2):
    # top, left, bottom, right = bounding_box
    draw = ImageDraw.Draw(image)
    for bbox in bounding_boxs:
        xmin, ymin, xmax, ymax = bbox
        for j in range(thickness):
            draw.rectangle([xmin+j, ymin+j, xmax-j, ymax-j], outline=(255, 0, 0))
    del draw 
    return image
    
    
    # 