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
    pose: list = None
    segment_class: str = None
    segment_object: str = None 
    ground_truth: list = None 
    
    def __repr__(self):
        s = []
        if self.path is not None:
            s.append(f"path='{self.path}'")
        if self.pil is not None:
            s.append(f"pil=PILImage({self.pil.size})")
        if self.label is not None:
            s.append(f"label={self.label}")
        if self.bound_box is not None:
            s.append(f"bound_box={self.bound_box}")
        if self.pose is not None:
            s.append(f"pose={self.pose}")
        if self.segment_class is not None:
            s.append(f"segement_class='{self.segment_class}'")
        if self.segment_object is not None:
            s.append(f"segment_object='{self.segment_object}'")
        if self.ground_truth is not None:
            s.append(f"ground_truth={self.ground_truth}")
        return f"ImageSource({','.join(s)})"
        
        



class ImageProcesser:
    
    def __init__(self, base_directory, output_directory="augment", class_label=None, n_job=1, 
                 fixed_size=None, save_to_disk=False, save_format=None):
        self.base_directory = os.path.abspath(base_directory)
        if not (os.path.exists(self.base_directory) and os.path.isdir(self.base_directory)):
            raise IOError(f"there is no such directory named {self.base_directory}.")
        self.output_directory = os.path.join(self.base_directory, output_directory)
        if not os.path.exists(self.output_directory) or (not os.path.isdir(self.output_directory)):
            try:
                os.mkdir(output_directory)
            except Exception:
                raise IOError("no authority to make a output directory.")
        self.class_label = class_label
        self.labels = set()
        self.operations = []
        self.fixed_size = fixed_size
        self.save_format = save_format or "jpg"
        self.save_to_disk = save_to_disk
        self.n_job = n_job
        self.image_sources = self.scan_directory(self.base_directory)
        
                
    def scan_directory(self, base_directory):
        image_sources = []
        for directory in glob.glob(os.path.join(base_directory, "*")):
            if directory == self.output_directory:
                continue
            label = os.path.basename(directory)
            self.labels.add(label)
            file_list = self.scan_images(directory)
            for file in file_list:
                image_sources.append(ImageSource(path=file, label=label))
        return image_sources
                
    def _check_outdir(self):
        for label in self.labels:
            path = os.path.join(self.output_directory, label)
            if not os.path.exists(path) or not os.path.isdir(path):
                try:
                    os.mkdir(path)
                except IOError as e:
                    print(f"failed to create output director of label '{label}' with {e.message}")
                
    @staticmethod
    def scan_images(direcotry):
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
    
    
    def _execute(self, image_source, save_to_disk=False):
        if image_source.path is not None:
            image = Image.open(image_source.path)
        elif image_source.pil is not None:
            image = image_source.pil
        else:
            raise ValueError("no image data.")
        # if image_source.ground_truth is not None:
        #     if isinstance(image_source.ground_truth, list):
        #         for image in image_source.ground_truth:
        #             images.append(Image.open(image))
        #     else:
        #         images.append(Image.open(image_source.ground_truth))
        for operation in self.operations:
            r = np.random.uniform(0, 1)
            if r < operation.p:
                image = operation.perform(image)
                break
        image = self.resize_to_fixed(image)
        if save_to_disk:
            file_name = str(uuid.uuid4())
            try:
                save_name = f"{image_source.label}/{file_name}.{self.save_format}"
                image.save(os.path.join(self.output_directory, save_name))
            except IOError as e:
                print("Error writing %s, %s. Change save_format to PNG?" % (file_name, e.message))
                print("You can change the save format using the set_save_format(save_format) function.")
                print("By passing save_format=\"auto\", Augmentor can save in the correct format automatically.")
        if self.class_label is None:
            label = image_source.label
        else:
            label = self.class_label.get(image_source)
        return image, label
    
    def __repr__(self):
        return f"ImageDataset(image:{len(self.image_sources)}, operation: {len(self.operations)})"
        
    def __str__(self):
        labels = ",".join([label for label in self.labels])
        return f"Image Dataset with label:\n\tImages: {len(self.image_sources)}\n\tOperations: {len(self.operaions)}\nLabels: {labels}"
            

    def process(self, sample=0):
        if len(self.image_sources) == 0:
            raise IndexError("no images in this dataset.")
        if len(self.operations) == 0:
            warnings.warn("no augment operations, origin images will be push out.")
        if sample == 0:
            image_sources = self.image_sources
            np.random.shuffle(image_sources)
        else:
            image_sources = [random.choice() for _ in range(sample)]
        n_job = int(self.n_job)
        images = []
        labels = []
        if self.save_to_disk:
            self._check_outdir()
        if n_job == 1:
            with tqdm(total=len(image_sources), desc="generating image data", unit=" samples") as progress_bar:
                for image_source in image_sources:
                    image_pil, image_label = self._execute(image_source, self.save_to_disk)
                    images.append(image_pil)
                    labels.append(image_label)
                    progress_bar.set_description(f"Processing {os.path.basename(image_source.path)}")
                    progress_bar.update(1)
        elif n_job > 1:
            with tqdm(total=len(image_sources), desc="generating image data", unit=" samples") as progress_bar:
                with ThreadPoolExecutor(max_workers=n_job) as executor:
                    for image_pil, image_label in executor.map(self._execute, image_sources, self.save_to_disk):
                        images.append(image_pil)
                        labels.append(image_label)
                        progress_bar.set_description(f"Processing with multi process")
                        progress_bar.update(1)
        
        return images, labels
    
    def add_operation(self, operation):
        self.operations.append(operation)
        
    def remove_operation(self, operation_index=-1):
        self.operations.pop(operation_index)
    
    def resize_to_fixed(self, pil):
        if self.fixed_size is None:
            return pil
        w, h = pil.size
        iw, ih = self.fixed_size
        scale = min(iw / w, ih / h)
        new_w, new_h = round(scale * w), round(scale * h)
        resized = pil.resize((new_w, new_h), Image.Resampling.BICUBIC)
        new_pil = Image.new(mode=pil.mode, size=self.fixed_size)
        new_pil.paste(resized, ((iw - new_w) // 2, (ih - new_h) // 2))
        return new_pil
        
        
    def generator(self, batch_size):
        if len(self.image_sources) == 0:
            raise IndexError("no images in this dataset.")
        if len(self.operations) == 0:
            warnings.warn("no augment operations, origin images will be push out.")
        n_job = int(self.n_job)
        np.random.shuffle(self.image_sources)
        batch_num = len(self.image_sources) // batch_size
        for i in range(batch_num):
            batch_images, batch_labels = [], []
            for _ in range(batch_size):
                image_sources = self.image_sources[i * batch_size: (i + 1) * batch_size]
                if n_job == 1:
                    for image_source in image_sources:
                        image_pil, image_label = self._execute(image_source)
                        batch_images.append(image_pil)
                        batch_labels.append(image_label)
                elif n_job > 1:
                    with ThreadPoolExecutor(max_workers=n_job) as executor:
                        for image_pil, image_label in executor.map(self._execute, image_sources):
                            batch_images.append(image_pil)
                            batch_labels.append(image_label)
            yield batch_images, batch_labels
        
            
    
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
    
    
    def __init__(self, base_directory, output_directory="augment", label_type="bndbox", n_job=1, 
                 skip_difficult=True, class_label=None, fixed_size=None, save_to_disk=False, save_format=None):
        if label_type not in ("bndbox", "pose", "segement_class", "segment_object"):
            raise ValueError('''label type should be in one of ("pose", "bndbox", 
                             "segement class", "segment object").''')
        self.label_type = label_type
        self.base_directory = os.path.abspath(base_directory)
        if not (os.path.exists(self.base_directory) and os.path.isdir(self.base_directory)):
            raise IOError(f"there is no such directory named {self.base_directory}.")
        self.output_directory = os.path.join(self.base_directory, output_directory)
        if not os.path.exists(self.output_directory) or (not os.path.isdir(self.output_directory)):
            try:
                os.mkdir(self.output_directory)
            except Exception:
                raise IOError("no authority to make a output directory.")
        self.class_label = class_label
        self.labels = set()
        self.operations = []
        self.fixed_size = fixed_size
        self.save_format = save_format or "jpg"
        self.skip_difficult = skip_difficult
        self.save_to_disk = save_to_disk
        self.n_job = n_job
        self.image_sources = self.scan_annotation(os.path.join(self.base_directory, self._voc_annotation))
    
    
    def scan_annotation(self, annotation_directory):
        image_sources = []
        jpg_directory = os.path.join(self.base_directory, self._voc_images)
        
        
        for xml_file in glob.glob(os.path.join(annotation_directory, "*.xml")):
            try:
                with open(xml_file, encoding='utf-8') as file:
                    tree = ET.parse(file)
            except IOError as e:
                print(f"can't read the file {xml_file}: {e.message}.")
                continue
            root = tree.getroot()
            image_file = root.find("filename").text
            image_path = os.path.join(jpg_directory, image_file)
            if self.label_type == "bndbox":
                bnd_boxes = []
                for obj in root.iter("object"):
                    if obj.find("difficult") is None:
                        difficult = 0
                    else:
                        difficult = int(obj.find("difficult").text)
                    if self.skip_difficult and difficult == 1:
                        continue
                    bnd_label = obj.find("name").text
                    xml_box = obj.find('bndbox')
                    box = (int(xml_box.find('xmin').text), int(xml_box.find('ymin').text), 
                            int(xml_box.find('xmax').text), int(xml_box.find('ymax').text))
                    bnd_boxes.append([bnd_label, box])
                if len(bnd_boxes) > 0:
                    image_sources.append(ImageSource(path=image_path, bound_box=bnd_boxes))
            elif self.label_type == "pose":
                pose_parts = []
                for obj in root.iter("object"):
                    if obj.find("pose") is None:
                        continue
                    pose = obj.find("pose").text
                    if obj.find("difficult") is None:
                        difficult = 0
                    else:
                        difficult = int(obj.find("difficult").text)
                    if self.skip_difficult and difficult == 1:
                        continue
                    parts = []
                    for part in obj.iter("part"):
                        part_label = part.find("name").text
                        box_xml = part.find("bndbox")
                        part_box = (int(box_xml.find('xmin').text), int(box_xml.find('ymin').text), 
                                    int(box_xml.find('xmax').text), int(box_xml.find('ymax').text))
                        parts.append([part_label, part_box])
                    if len(parts) > 0:
                        pose_parts.append([pose, parts])
                if len(pose_parts) > 0:
                    image_sources.append(ImageSource(path=image_path, pose=pose_parts))     
            else:
                segemented = root.find("segmented")
                if segemented is None or int(segemented.text) == 0:
                    continue
                if self.label_type == "segement_class":
                    class_dir = os.path.join(self.base_directory, self._voc_segclass)
                    class_path = os.path.join(class_dir, image_file)
                    image_sources.append(ImageSource(path=image_path, segment_class=class_path))
                else:
                    object_dir = os.path.join(self.base_directory, self._voc_segobject)
                    object_path = os.path.join(object_dir, image_file)
                    image_sources.append(ImageSource(path=image_path, segement_object=object_path))
                
        return image_sources

    
    def __repr__(self):
        return f"VOCDataset(image:{len(self.image_sources)}, operation: {len(self.operations)})"
        
    def __str__(self):
        labels = ",".join([label for label in self.labels])
        return f"Image Dataset of VOC:\n\tImages: {len(self.image_sources)}\n\tOperations: {len(self.operations)}\n\tLabels: {labels}"
        
    
    def _check_outdir(self):
        ...
    
    def _execute(self, image_source, save_to_disk=False):
        if image_source.path is not None:
            image = Image.open(image_source.path)
        elif image_source.pil is not None:
            image = image_source.pil
        else:
            raise ValueError("no image data.")
        # if image_source.ground_truth is not None:
        #     if isinstance(image_source.ground_truth, list):
        #         for image in image_source.ground_truth:
        #             images.append(Image.open(image))
        #     else:
        #         images.append(Image.open(image_source.ground_truth))
        for operation in self.operations:
            r = np.random.uniform(0, 1)
            if r < operation.p:
                image = operation.perform(image)
                break
        image = self.resize_to_fixed(image)
        if save_to_disk:
            file_name = str(uuid.uuid4())
            try:
                save_name = f"{image_source.label}/{file_name}.{self.save_format}"
                image.save(os.path.join(self.output_directory, save_name))
            except IOError as e:
                print("Error writing %s, %s. Change save_format to PNG?" % (file_name, e.message))
                print("You can change the save format using the set_save_format(save_format) function.")
                print("By passing save_format=\"auto\", Augmentor can save in the correct format automatically.")
        if self.class_label is None:
            label = image_source.label
        else:
            label = self.class_label.get(image_source)
        return image, label
    
    def __repr__(self):
        return f"ImageDataset(image:{len(self.image_sources)}, operation: {len(self.operations)})"
        
    def __str__(self):
        labels = ",".join([label for label in self.labels])
        return f"Image Dataset with label:\n\tImages: {len(self.image_sources)}\n\tOperations: {len(self.operaions)}\nLabels: {labels}"
            

    def process(self, sample=0):
        if len(self.image_sources) == 0:
            raise IndexError("no images in this dataset.")
        if len(self.operations) == 0:
            warnings.warn("no augment operations, origin images will be push out.")
        if sample == 0:
            image_sources = self.image_sources
            np.random.shuffle(image_sources)
        else:
            image_sources = [random.choice() for _ in range(sample)]
        n_job = int(self.n_job)
        images = []
        labels = []
        if self.save_to_disk:
            self._check_outdir()
        if n_job == 1:
            with tqdm(total=len(image_sources), desc="generating image data", unit=" samples") as progress_bar:
                for image_source in image_sources:
                    image_pil, image_label = self._execute(image_source, self.save_to_disk)
                    images.append(image_pil)
                    labels.append(image_label)
                    progress_bar.set_description(f"Processing {os.path.basename(image_source.path)}")
                    progress_bar.update(1)
        elif n_job > 1:
            with tqdm(total=len(image_sources), desc="generating image data", unit=" samples") as progress_bar:
                with ThreadPoolExecutor(max_workers=n_job) as executor:
                    for image_pil, image_label in executor.map(self._execute, image_sources, self.save_to_disk):
                        images.append(image_pil)
                        labels.append(image_label)
                        progress_bar.set_description(f"Processing with multi process")
                        progress_bar.update(1)
        
        return images, labels
    
    def add_operation(self, operation):
        self.operations.append(operation)
        
    def remove_operation(self, operation_index=-1):
        self.operations.pop(operation_index)
    
    def resize_to_fixed(self, pil):
        if self.fixed_size is None:
            return pil
        w, h = pil.size
        iw, ih = self.fixed_size
        scale = min(iw / w, ih / h)
        new_w, new_h = round(scale * w), round(scale * h)
        resized = pil.resize((new_w, new_h), Image.Resampling.BICUBIC)
        new_pil = Image.new(mode=pil.mode, size=self.fixed_size)
        new_pil.paste(resized, ((iw - new_w) // 2, (ih - new_h) // 2))
        return new_pil
    
    def generator(self, batch_size):
        if len(self.image_sources) == 0:
            raise IndexError("no images in this dataset.")
        if len(self.operations) == 0:
            warnings.warn("no augment operations, origin images will be push out.")
        n_job = int(self.n_job)
        np.random.shuffle(self.image_sources)
        batch_num = len(self.image_sources) // batch_size
        for i in range(batch_num):
            batch_images, batch_labels = [], []
            for _ in range(batch_size):
                image_sources = self.image_sources[i * batch_size: (i + 1) * batch_size]
                if n_job == 1:
                    for image_source in image_sources:
                        image_pil, image_label = self._execute(image_source)
                        batch_images.append(image_pil)
                        batch_labels.append(image_label)
                elif n_job > 1:
                    with ThreadPoolExecutor(max_workers=n_job) as executor:
                        for image_pil, image_label in executor.map(self._execute, image_sources):
                            batch_images.append(image_pil)
                            batch_labels.append(image_label)
            yield batch_images, batch_labels
            
            
            
            
        
            
        