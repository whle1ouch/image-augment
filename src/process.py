from dataclasses import dataclass
import os, glob, uuid, warnings, random
from PIL import Image
from tqdm import tqdm
from xml.etree import ElementTree as ET
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing_extensions import Literal
from .operation import PixelOperation



@dataclass
class ImageSource:
    
    path: str = None
    pil: Image = None
    label: str = None
    bounding_box: list = None
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
        if self.bounding_box is not None:
            s.append(f"bounding_box={self.bounding_box}")
        if self.pose is not None:
            s.append(f"pose={self.pose}")
        if self.segment_class is not None:
            s.append(f"segement_class='{self.segment_class}'")
        if self.segment_object is not None:
            s.append(f"segment_object='{self.segment_object}'")
        if self.ground_truth is not None:
            s.append(f"ground_truth={self.ground_truth}")
        return f"ImageSource({','.join(s)})"


DatasetType = Literal["label", "object", "pose", "segment_class", "segment_object"]
LABEL:DatasetType = "label"
OBJECT:DatasetType = "object"
POSE:DatasetType = "pose"
SEGMENT_CLASS:DatasetType = "segment_class"
SEGMENT_OBJECT:DatasetType = "segment_object"
        
_voc_annotation = "Annotations"
_voc_images = "JPEGImages"
_voc_segclass = "SegmentationClass"
_voc_segobject = "SegmentationObject"


class ImageDataset:
    
    
    
    def __init__(self, output_directory=None, class_label=None, n_job=1,
                 fixed_size=None, save_to_disk=False, save_format=None):        
        self.output_directory = output_directory
        self.class_label = class_label
        self._target = set()
        self.operations = list()
        self.fixed_size = fixed_size
        self.save_format = save_format or "jpg"
        self.save_to_disk = save_to_disk
        self.n_job = n_job
        self.output_target_file = None
        self.image_sources = list()
        

    def scan_directory(self, base_directory):
        if not (os.path.exists(base_directory) and os.path.isdir(base_directory)):
            raise IOError(f"there is no such directory named {base_directory}.")
        if self.target_type != "label":
            raise ValueError(f"'scan_directory only can be used for label type dataset'")
        image_sources = []
        for directory in glob.glob(os.path.join(base_directory, "*")):
            if directory == self.output_directory:
                continue
            label = os.path.basename(directory)
            self._target.add(label)
            file_list = self.scan_images(directory)
            for file in file_list:
                image_sources.append(ImageSource(path=file, label=label))
        self.image_sources =  image_sources
    
        
    
    def scan_voc(self, base_directory, skip_difficult=True):
        if not (os.path.exists(base_directory) and os.path.isdir(base_directory)):
            raise IOError(f"there is no such directory named {base_directory}.")
        image_sources = list()
        annotation_directory = os.path.join(base_directory, self._voc_annotation)
        jpg_directory = os.path.join(base_directory, self._voc_images)
        class_directory = os.path.join(base_directory, self._voc_segclass)
        object_directory = os.path.join(base_directory, self._voc_segobject)
        
        
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
            if self.target_type == "label":
                obj_list = root.findall("object")
                if len(obj_list) != 1:
                    continue
                obj = obj_list[0]
                if obj.find("difficult") is None:
                    difficult = 0
                else:
                    difficult = int(obj.find("difficult").text)
                if skip_difficult and difficult == 1:
                        continue
                obj_label = obj.find("name").text
                self._target.add(obj_label)
                image_sources.append(ImageSource(path=image_path, label=obj_label))
            elif self.target_type == "bounding_box":
                bnd_boxes = list()
                for obj in root.iter("object"):
                    if obj.find("difficult") is None:
                        difficult = 0
                    else:
                        difficult = int(obj.find("difficult").text)
                    if skip_difficult and difficult == 1:
                        continue
                    bnd_label = obj.find("name").text
                    self._target.add(bnd_label)
                    xml_box = obj.find('bndbox')
                    box = (int(xml_box.find('xmin').text), int(xml_box.find('ymin').text), 
                            int(xml_box.find('xmax').text), int(xml_box.find('ymax').text))
                    bnd_boxes.append([bnd_label, box])
                if len(bnd_boxes) > 0:
                    image_sources.append(ImageSource(path=image_path, bounding_box=bnd_boxes))
            elif self.target_type == "pose":
                pose_parts = list()
                for obj in root.iter("object"):
                    if obj.find("pose") is None:
                        continue
                    pose = obj.find("pose").text
                    self._target.add(pose)
                    if obj.find("difficult") is None:
                        difficult = 0
                    else:
                        difficult = int(obj.find("difficult").text)
                    if skip_difficult and difficult == 1:
                        continue
                    parts = list()
                    for part in obj.iter("part"):
                        part_label = part.find("name").text
                        self._target.add(part_label)
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
                if self.target_type == "segement_class":
                    class_path = os.path.join(class_directory, image_file)
                    image_sources.append(ImageSource(path=image_path, segment_class=class_path))
                else:
                    
                    object_path = os.path.join(object_directory, image_file)
                    image_sources.append(ImageSource(path=image_path, segement_object=object_path))
        self.image_sources =  image_sources
        
    def load(self, base_directory):
        base_path = os.path.abspath(base_directory)
        if not os.path.exists(base_path) or (not os.path.isdir(base_path)):
            raise ValueError(f"{base_path} is not existed or is not a directory")
        label_path = os.path.join(base_path, "label")
        image_sources = list()
        if self.target_type in ("pose", "label", "bounding_box"):
            label_file = os.path.join(label_path, 'label.txt')
            with open(label_file, "r", encoding="utf-8") as file:
                lines = file.readlines()
            for line in lines:
                try:
                    image_path, target = self._parse(line)
                    if len(image_path) > 0 and len(target) > 0:
                        image_source = ImageSource(path=image_path)
                        image_source.__setattr__(self.target_type, target)
                        image_sources.append(image_source)
                except Exception as e:
                    print(f"fail to read line as '{line}': {e.args}.")
        else:
            image_files = set([os.path.basename(path) for path in  self.scan_images(base_path)])
            label_files = set([os.path.basename(path) for path in  self.scan_images(label_path)])
            valid_files = image_files & label_files
            for file in valid_files:
                image_source = ImageSource(path=os.path.join(base_path, file))
                image_source.__setattr__(self.target_type, os.path.join(label_path, file))
                image_sources.append(image_source)
        self.image_sources = image_sources
                
        
    def set_output_directory(self, directory):
        try:
            self.output_directory = os.path.abspath(directory)
        except TypeError as e :
            print(f"fail to set the director of '{directory}', {e.message}")
        
                
    def _check_output_directory(self):
        if self.output_directory is None:
            raise ValueError("no output direcotry to save iamge, use 'set_output_directory' firstly.")
        if not os.path.exists(self.output_directory) or (not os.path.isdir(self.output_directory)):
            try:
                os.mkdir(self.output_directory)
            except Exception:
                raise IOError("no authority to make a output directory.")
        target_path = os.path.join(self.output_directory, "label")
        for path in [self.output_directory, target_path]:
            if not os.path.exists(path) or not os.path.isdir(path):
                try:
                    os.mkdir(path)
                except IOError as e:
                    print(f"failed to create output target director of '{path}': {e.message}")
        self.output_target_file = os.path.join(target_path, "label.txt")
        if os.path.exists(self.output_target_file):
            warnings.warn("label text file has already existed, process result may be appended from last lines.")
                
    
    
    def save_image(self, image, target):
        file_name = str(uuid.uuid4())
        try:
            save_name = f"{file_name}.{self.save_format}"
            image.save(os.path.join(self.output_directory, save_name))
            if self.target_type in ("segement_class", "segement_object"):
                target_name = f"target/{file_name}.{self.save_format}"
                target.save(os.path.join(self.output_directory, target_name))
            else:
                ts = self._stringify(target, save_name)
                with open(self.output_target_file, mode="a", encoding="utf-8") as file:
                    file.write(ts)
        except IOError as e:
            print("Error writing %s, %s. Change save_format to PNG?" % (file_name, e.message))
            print("You can change the save format using the set_save_format(save_format) function.")
            print("By passing save_format=\"auto\", Augmentor can save in the correct format automatically.")
    
    
    def _stringify(self, target, save_name):
        if self.target_type == "label":
            return " ".join(["\n", save_name, target])
        elif self.target_type == "bounding_box":
            st= [save_name]
            for label, bbox in target:
                st.append(str(label))
                st += [str(b) for b in bbox]
            return "\n" + " ".join(st)
        elif self.target_type == "pose":
            st = [save_name]
            for pose, pose_parts in target:
                s = [pose]
                for part, bbox in pose_parts:
                    s.append(str(part))
                    s += [str(b) for b in bbox]
                st.append("/".join(s))
            return "\n" + " ".join(st)
        else:
            raise AttributeError(f"target of {self.target_type} can not be stringfied")
            
    
    def parse_target(self, items):
        if self.target_type == "label":
            target = items[0]
        elif self.target_type == "bounding_box":
            target = []
            n = len(items) // 5
            for i in range(n):
                target.append([items[i*5], (int(items[i*5+1]), int(items[i*5+2]),
                                            int(items[i*5+3]), int(items[i*5+4]))])
        elif self.target_type == "pose":
            target = []
            for pose_string in items:
                pose_items = pose_string.split("/")
                pose_label = pose_items[0]
                part_list = pose_items[1:]
                part_num = len(part_list) // 5
                pose_parts = []
                for i in range(part_num):
                    part_label = part_list[i * 5]
                    part_box = (int(part_list[i*5+1]), int(part_list[i*5+2]), int(part_list[i*5+3]), int(part_list[i*5+4]))
                    pose_parts.append([part_label, part_box])
                print(pose_parts)
                target.append([pose_label, pose_parts])
        else:
            raise AttributeError(f"lines can not be parsed as {self.target_type}")
        return target
                    
    
    
    
    def _execute(self, image_source, save_to_disk=False):
        if image_source.path is not None:
            image = Image.open(image_source.path)
        elif image_source.pil is not None:
            image = image_source.pil
        else:
            raise ValueError("no image data.")
        target = self.get_target(image_source)
        image, target = self._random_operation(image, target)
        image = self.resize_to_fixed(image)
        if self.target_type in ("segment_class", "segement_object"):
            target = self.resize_to_fixed(target)
        if save_to_disk:
            self.save_image(image, target)
        return image, target
    
    def get_target(self, image_source):
        target = image_source.__dict__.get(self.target_type, None)
        if target is None:
            raise IndexError(f"image source '{image_source}' has no {self.target_type}")
        
        if self.class_label is not None:
            if self.target_type == "label":
                target = self.class_label.get(target)
            elif self.target_type == "bounding_box":
                target =  [[self.class_label.get(label), bbox] for label, bbox in target]
            elif self.target_type == "pose":
                for i in range(len(target)):
                    target[i][0] = self.class_label.get(target[i][0])
                    pose_parts = target[i][1]
                    for j in range(len(pose_parts)):
                        pose_parts[j][0] = self.class_label.get(pose_parts[j][0])
                    target[i][1] = pose_parts
        if self.target_type in ("segement_class", "segment_object"):
            target = Image.open(target)
        return target
    
    def _random_operation(self, image, target):
        for operation in self.operations:
            r = np.random.uniform(0, 1)
            if r < operation.p:
                if isinstance(operation, PixelOperation) or self.target_type == "label":
                    image = operation.perform(image)
                    target = target
                elif self.target_type == "bounding_box":
                    ...
                elif self.target_type == "pose":
                    ...
                else:
                    ...
                break
        return image, target 
        
        
    def __repr__(self):
        return f"ImageDataset(image:{len(self.image_sources)}, operation: {len(self.operations)})"
        
    def __str__(self):
        target_string = ",".join([t for t in self._target])
        s = f"Image Dataset Processer with target of {self.target_type}:\n\tImages: {len(self.image_sources)}\n\tOperations: {len(self.operations)}\n\tLabels: {target_string}"
        if self.fixed_size is not None:
            s += f"\n\tImage Size: {self.fixed_size}"
        if self.output_directory is not None:
            s += f"\n\tSave Path: {self.output_directory}\n\tSave Format: {self.save_format}"
        return s
        
            

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
        images = list()
        targets = list()
        if self.save_to_disk:
            self._check_output_directory()
        if n_job == 1:
            with tqdm(total=len(image_sources), desc="generating image data", unit=" samples") as progress_bar:
                for image_source in image_sources:
                    image_pil, image_target = self._execute(image_source, self.save_to_disk)
                    images.append(image_pil)
                    targets.append(image_target)
                    progress_bar.set_description(f"Processing {os.path.basename(image_source.path)}")
                    progress_bar.update(1)
        elif n_job > 1:
            with tqdm(total=len(image_sources), desc="generating image data", unit=" samples") as progress_bar:
                with ThreadPoolExecutor(max_workers=n_job) as executor:
                    for image_pil, image_target in executor.map(self._execute, image_sources, self.save_to_disk):
                        images.append(image_pil)
                        targets.append(image_target)
                        progress_bar.set_description(f"Processing with multi process")
                        progress_bar.update(1)
        
        return images, targets
    
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
        
                
                
class LabelImageDataset(ImageDataset):
    
    
    def scan_directory(self, base_directory):
        if not (os.path.exists(base_directory) and os.path.isdir(base_directory)):
            raise IOError(f"there is no such directory named {base_directory}.")
        image_sources = []
        for directory in glob.glob(os.path.join(base_directory, "*")):
            label = os.path.basename(directory)
            
            file_list = self.scan_images(directory)
            if len(file_list) > 0:
                self._target.add(label)
            for file in file_list:
                image_sources.append(ImageSource(path=file, label=label))
        self.image_sources =  image_sources
    
        
    
    def scan_voc(self, base_directory, skip_difficult=True):
        if not (os.path.exists(base_directory) and os.path.isdir(base_directory)):
            raise IOError(f"there is no such directory named {base_directory}.")
        image_sources = list()
        annotation_directory = os.path.join(base_directory, self._voc_annotation)
        jpg_directory = os.path.join(base_directory, self._voc_images)

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
            obj_list = root.findall("object")
            if len(obj_list) != 1:
                continue
            obj = obj_list[0]
            if obj.find("difficult") is None:
                difficult = 0
            else:
                difficult = int(obj.find("difficult").text)
            if skip_difficult and difficult == 1:
                    continue
            obj_label = obj.find("name").text
            self._target.add(obj_label)
            image_sources.append(ImageSource(path=image_path, label=obj_label))
           
        self.image_sources =  image_sources
    

class ObjectImageDataset(ImageDataset):
    
    def scan_voc(self, base_directory, skip_difficult=True):
        if not (os.path.exists(base_directory) and os.path.isdir(base_directory)):
            raise IOError(f"there is no such directory named {base_directory}.")
        image_sources = list()
        annotation_directory = os.path.join(base_directory, self._voc_annotation)
        jpg_directory = os.path.join(base_directory, self._voc_images)
        
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
            bnd_boxes = list()
            for obj in root.iter("object"):
                if obj.find("difficult") is None:
                    difficult = 0
                else:
                    difficult = int(obj.find("difficult").text)
                if skip_difficult and difficult == 1:
                    continue
                bnd_label = obj.find("name").text
                self._target.add(bnd_label)
                xml_box = obj.find('bndbox')
                box = (int(xml_box.find('xmin').text), int(xml_box.find('ymin').text), 
                        int(xml_box.find('xmax').text), int(xml_box.find('ymax').text))
                bnd_boxes.append([bnd_label, box])
            if len(bnd_boxes) > 0:
                image_sources.append(ImageSource(path=image_path, bounding_box=bnd_boxes))
        self.image_sources =  image_sources
    


class PoseImageDataset(ImageDataset):
    
    def scan_voc(self, base_directory, skip_difficult=True):
        if not (os.path.exists(base_directory) and os.path.isdir(base_directory)):
            raise IOError(f"there is no such directory named {base_directory}.")
        image_sources = list()
        annotation_directory = os.path.join(base_directory, self._voc_annotation)
        jpg_directory = os.path.join(base_directory, self._voc_images)
        
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
            pose_parts = list()
            for obj in root.iter("object"):
                if obj.find("pose") is None:
                    continue
                pose = obj.find("pose").text
                self._target.add(pose)
                if obj.find("difficult") is None:
                    difficult = 0
                else:
                    difficult = int(obj.find("difficult").text)
                if skip_difficult and difficult == 1:
                    continue
                parts = list()
                for part in obj.iter("part"):
                    part_label = part.find("name").text
                    self._target.add(part_label)
                    box_xml = part.find("bndbox")
                    part_box = (int(box_xml.find('xmin').text), int(box_xml.find('ymin').text), 
                                int(box_xml.find('xmax').text), int(box_xml.find('ymax').text))
                    parts.append([part_label, part_box])
                if len(parts) > 0:
                    pose_parts.append([pose, parts])
            if len(pose_parts) > 0:
                image_sources.append(ImageSource(path=image_path, pose=pose_parts))     
        self.image_sources =  image_sources
    
    
    
class SegmentImageDataset(ImageDataset):
    
    def scan_voc(self, base_directory, skip_difficult=True):
        if not (os.path.exists(base_directory) and os.path.isdir(base_directory)):
            raise IOError(f"there is no such directory named {base_directory}.")
        image_sources = list()
        annotation_directory = os.path.join(base_directory, self._voc_annotation)
        jpg_directory = os.path.join(base_directory, self._voc_images)
        class_directory = os.path.join(base_directory, self._voc_segclass)
        object_directory = os.path.join(base_directory, self._voc_segobject)
        
        
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
            segemented = root.find("segmented")
            if segemented is None or int(segemented.text) == 0:
                continue
            if self.target_type == "segement_class":
                class_path = os.path.join(class_directory, image_file)
                image_sources.append(ImageSource(path=image_path, segment_class=class_path))
            else:
                
                object_path = os.path.join(object_directory, image_file)
                image_sources.append(ImageSource(path=image_path, segement_object=object_path))
        self.image_sources =  image_sources
    