from dataclasses import dataclass
import os, glob, uuid, warnings, random
from PIL import Image
from tqdm import tqdm
from xml.etree import ElementTree as ET
import numpy as np
from concurrent.futures import ThreadPoolExecutor


@dataclass
class ImageSource:
    
    path: str = None
    pil: str = None
    label: str = None
    bound_box: list = None
    segment_class: str = None
    segement_object: str = None 
    ground_truth: list = None 



class ImageProcesser:
    
    def __init__(self, base_directory, output_directory="augment", class_label=None, sample_rate=0, save_to_disk=False, save_format=None, n_job=1):
        self.base_directory = os.path.abspath(base_directory)
        self.output_directory = os.path.join(self.base_directory, output_directory)
        self.class_label = class_label
        self.labels = set()
        self.image_sources = []
        self.operations = []
        self.save_format = save_format or "jpg"
        self.save_to_disk = save_to_disk
        self.n_job = n_job
        self.sample_rate = sample_rate
        self._check_image()
        
    def _check_image(self):
        if not (os.path.exists(self.base_directory) and os.path.isdir(self.base_directory)):
            raise IOError(f"there is no such directory named {self.base_directory}.")
        if not os.path.exists(self.output_directory):
            try:
                os.mkdir(self.output_directory)
            except Exception:
                raise IOError("no authority to make a output directory.")
        elif not os.path.isdir(self.output_directory):
            raise IOError(f"{self.output_directory} exists and  is not a directory.")
            
        for directory in glob.glob(os.path.join(self.base_directory, "*")):
            if directory == self.output_directory:
                continue
            label = os.path.basename(directory)
            self.labels.add(label)
            file_list = self.scan_directory(directory)
            for file in file_list:
                self.image_sources.append(ImageSource(path=file, label=label))
                
    def _check_outdir(self):
        for label in self.labels:
            path = os.path.join(self.output_directory, label)
            if not os.path.exists(path) or not os.path.isdir(path):
                try:
                    os.mkdir(path)
                except IOError as e:
                    print(f"failed to create output director of label '{label}' with {e.message}")
                
    @staticmethod
    def scan_directory(direcotry):
        file_types = ['*.jpg', '*.bmp', '*.jpeg', '*.gif', '*.img', '*.png', '*.tiff', '*.tif']
        list_of_files = []

        if os.name == "nt":
            for file_type in file_types:
                list_of_files.extend(glob.glob(os.path.join(os.path.abspath(direcotry), file_type)))
        else:
            file_types.extend([str.upper(str(x)) for x in file_types])
            for file_type in file_types:
                list_of_files.extend(glob.glob(os.path.join(os.path.abspath(direcotry), file_type)))
        return list_of_files
    
    
    def _execute(self, image_source):
        images = []
        if image_source.path is not None:
            images.append(Image.open(image_source.path))
        elif image_source.pil is not None:
            images.append(image_source.pil)
        else:
            raise ValueError("no image data.")
        if image_source.ground_truth is not None:
            if isinstance(image_source.ground_truth, list):
                for image in image_source.ground_truth:
                    images.append(Image.open(image))
            else:
                images.append(Image.open(image_source.ground_truth))
        for i in range(len(images)):
            for operation in self.operations:
                r = np.random.uniform(0, 1)
                if r < operation.p:
                    images[i] = operation.perform(images[i])
                    break
        if self.save_to_disk:
            file_name = str(uuid.uuid4())
            try:
                for i in range(len(images)):
                    save_name = f"{image_source.label}/{'original' if i == 0 else 'groundtruth_(' + i +')'}_{file_name}.{self.save_format}"
                    images[i].save(os.path.join(self.output_directory, save_name))
            except IOError as e:
                print("Error writing %s, %s. Change save_format to PNG?" % (file_name, e.message))
                print("You can change the save format using the set_save_format(save_format) function.")
                print("By passing save_format=\"auto\", Augmentor can save in the correct format automatically.")
        return images, [image_source.label for _ in range(len(images))]
            

    def process(self):
        if len(self.image_sources) == 0:
            raise IndexError("no images in this dataset.")
        if len(self.operations) == 0:
            warnings.warn("no augment operations, origin images will be push out.")
        if self.sample_rate == 0:
            image_sources = self.image_sources
            np.random.shuffle(image_sources)
        else:
            n = int(np.ceil(self.sample_rate * len(self.image_sources)))
            image_sources = [random.choice() for _ in range(n)]
        n_job = int(self.n_job)
        images = []
        labels = []
        if self.save_to_disk:
            self._check_outdir()
        if n_job == 1:
            with tqdm(total=len(image_sources), desc="generating image data", unit=" samples") as progress_bar:
                for image_source in image_sources:
                    image_pils, image_labels = self._execute(image_source)
                    images.extend(image_pils)
                    labels.extend(labels)
                    progress_bar.set_description(f"Processing {os.path.basename(image_source.path)}")
                    progress_bar.update(1)
        elif n_job > 1:
            with tqdm(total=len(image_sources), desc="generating image data", unit=" samples") as progress_bar:
                with ThreadPoolExecutor(max_workers=n_job) as executor:
                    for img, label in executor.map(self._execute, image_sources):
                        images.extend(img)
                        labels.extend(label)
                        progress_bar.set_description(f"Processing with multi process")
                        progress_bar.update(1)
        
        return images, labels
    
    def add_operation(self, operation):
        self.operations.append(operation)
    
    @staticmethod
    def resize(image, size):
        w, h = image.size
        iw, ih = size
        scale = min(iw / w, ih / h)
        new_w, new_h = round(scale * w), round(scale * h)
        resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        new_image = Image.new(mode=image.mode, size=size)
        new_image.paste(resized, ((iw - new_w) // 2, (ih - new_h) // 2))
        return new_image
        
        
        

class VOCDataset:
    _voc_annotation = "Annotations"
    _voc_images = "JPEGImages"
    _voc_segclass = "SegmentationClass"
    _voc_segobject = "SegmentationObject"
    
    
    def __init__(self, label_type):
        if label_type not in ("bounding_box", "segement_class", "segment_object"):
            raise ValueError('''label type should be in one of ("class", "bound box", 
                             "segement class", "segment object").''')
        