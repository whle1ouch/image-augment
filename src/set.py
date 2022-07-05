from dataclasses import dataclass
import os, glob, uuid, warnings, random
from PIL import Image
from PIL.ImageFile import ImageFile
from matplotlib import image
from tqdm import tqdm
from xml.etree import ElementTree as ET
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing_extensions import Literal
from .operation import PixelOperation
from .utilies import scan_images



@dataclass
class ImageSource:
    
    path: str = None
    pil: Image = None
    label: str = None
    pose: list = None
    ground_truth: list = None
    bounding_box: list = None
    segment_class: str = None
    segment_object: str = None 
    
    def __repr__(self):
        s = []
        if self.path is not None:
            s.append(f"path='{self.path}'")
        if self.pil is not None:
            s.append(f"pil=PILImage({self.pil.size})")
        if self.label is not None:
            s.append(f"label={self.label}")
        if self.pose is not None:
            s.append(f"pose={self.pose}")
        if self.ground_truth is not None:
            s.append(f"ground_truth={self.ground_truth}")
        if self.bounding_box is not None:
            s.append(f"bounding_box={self.bounding_box}")
        if self.segment_class is not None:
            s.append(f"segement_class='{self.segment_class}'")
        if self.segment_object is not None:
            s.append(f"segment_object='{self.segment_object}'")
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


class LabelImageSet:
      
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
        self.output_target_path = None
        self.image_sources = list()
        
    def scan_directory(self, base_directory):
        if not (os.path.exists(base_directory) and os.path.isdir(base_directory)):
            raise IOError(f"there is no such directory named {base_directory}.")
        image_sources = []
        for directory in glob.glob(os.path.join(base_directory, "*")):
            label = os.path.basename(directory)
            
            file_list = scan_images(directory)
            if len(file_list) > 0:
                self._target.add(label)
            for file in file_list:
                image_sources.append(ImageSource(path=file, label=label))
        self.image_sources =  image_sources
    
    def scan_voc(self, base_directory, skip_difficult=True):
        if not (os.path.exists(base_directory) and os.path.isdir(base_directory)):
            raise IOError(f"there is no such directory named {base_directory}.")
        image_sources = list()
        annotation_directory = os.path.join(base_directory, _voc_annotation)
        jpg_directory = os.path.join(base_directory, _voc_images)

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

    def check_output_directory(self):
        if self.output_directory is None:
            raise ValueError("no output direcotry to save iamge, use 'set_output_directory' firstly.")
        if not os.path.exists(self.output_directory) or (not os.path.isdir(self.output_directory)):
            try:
                os.mkdir(self.output_directory)
            except Exception:
                raise IOError("no authority to make a output directory.")
        self.output_target_path = os.path.join(self.output_directory, "label.txt")
        if os.path.exists(self.output_target_path):
            warnings.warn("label text file has already existed, processing results may be appended at the end.")
                
    def save_image(self, image, target):
        file_name = str(uuid.uuid4())
        try:
            save_name = f"{file_name}.{self.save_format}"
            image.save(os.path.join(self.output_directory, save_name))
            rs = self.dump_target(target, save_name)
            with open(self.output_target_path, mode="a", encoding="utf-8") as file:
                file.write(rs)
        except IOError as e:
            print("Error writing %s, %s. Change save_format to PNG?" % (file_name, e.message))
            print("You can change the save format using the set_save_format(save_format) function.")
            print("By passing save_format=\"auto\", Augmentor can save in the correct format automatically.")
    
    
    def dump_target(self, target, save_name):
        return "\n" + " ".join([save_name, str(target)])
    
    def load(self, base_directory):
        base_path = os.path.abspath(base_directory)
        if not os.path.exists(base_path) or (not os.path.isdir(base_path)):
            raise ValueError(f"{base_path} is not existed or is not a directory")
        image_sources = list()
        label_file = os.path.join(base_directory, 'label.txt')
        with open(label_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
        for line in lines:
            try:
                items = [item.strip() for item in line.split(" ")]
                print(items)
                image_path, label = items[0], items[1]
                
                if len(image_path) > 0 and len(label) > 0:
                    image_source = ImageSource(path=image_path, label=label)
                    image_sources.append(image_source)
            except Exception as e:
                print(f"fail to read line as '{line}': {e.args}.")
        self.image_sources = image_sources
  
    def _execute(self, image_source, save_to_disk=False):
        if image_source.path is not None:
            image = Image.open(image_source.path)
        elif image_source.pil is not None:
            image = image_source.pil
        else:
            raise ValueError("no image data.")
        target = self.get_target(image_source)
        image, target = self._random_operation(image, target)
        image, target = self.resize_to_fixed(image, target)
        if save_to_disk:
            self.save_image(image, target)
        return image, target
    
    def get_target(self, image_source):
        target = image_source.label
        if target is None:
            raise IndexError(f"image source '{image_source}' has no {self.target_type}")
        if self.class_label is not None:
            target = self.class_label.get(target)
        return target
            
    def __repr__(self):
        return f"ImageDataset(image:{len(self.image_sources)}, operation: {len(self.operations)})"
        
    def __str__(self):
        target_string = ",".join([t for t in self._target])
        s = f"Image Dataset Processer with target :\n\tImages: {len(self.image_sources)}\n\tOperations: {len(self.operations)}\n\tLabels: {target_string}"
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
            self.check_output_directory()
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
    
    def _random_operation(self, image, target):
        for operation in self.operations:
            r = np.random.uniform(0, 1)
            if r < operation.probability:
                image = operation.perform(image)
                target = target
        return image, target 
    
    def add_operation(self, operation):
        self.operations.append(operation)
        
    def remove_operation(self, operation_index=-1):
        self.operations.pop(operation_index)
    
    def resize_to_fixed(self, pil, target):
        if self.fixed_size is None:
            return pil, target
        w, h = pil.size
        iw, ih = self.fixed_size
        scale = min(iw / w, ih / h)
        new_w, new_h = round(scale * w), round(scale * h)
        resized = pil.resize((new_w, new_h), Image.Resampling.BICUBIC)
        new_pil = Image.new(mode=pil.mode, size=self.fixed_size)
        new_pil.paste(resized, ((iw - new_w) // 2, (ih - new_h) // 2))
        return new_pil, target
        
        
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
        


class ObjectImageSet(LabelImageSet):
    
    def scan_voc(self, base_directory, skip_difficult=True):
        if not (os.path.exists(base_directory) and os.path.isdir(base_directory)):
            raise IOError(f"there is no such directory named {base_directory}.")
        image_sources = list()
        annotation_directory = os.path.join(base_directory, _voc_annotation)
        jpg_directory = os.path.join(base_directory, _voc_images)
        
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
            gt = list()
            bnd_boxes = list()
            index = 0
            for obj in root.iter("object"):
                if obj.find("difficult") is None:
                    difficult = 0
                else:
                    difficult = int(obj.find("difficult").text)
                if skip_difficult and difficult == 1:
                    continue
                bnd_label = obj.find("name").text
                xml_box = obj.find('bndbox')
                box = (int(xml_box.find('xmin').text), int(xml_box.find('ymin').text), 
                        int(xml_box.find('xmax').text), int(xml_box.find('ymax').text))
                gt.append(bnd_label)
                bnd_boxes.append(box)
                index += 1
            image_sources.append(ImageSource(path=image_path, ground_truth=gt, bounding_box=bnd_boxes))
        self.image_sources = image_sources
        
        
    def get_target(self, image_source):
        gts = image_source.ground_truth
        bnds = image_source.bounding_box
        if self.class_label is not None:
            gts = [self.class_label.get(gt) for gt in image_source.ground_truth]
        return gts, bnds

    
    def dump_target(self, target, save_name):
        st= [save_name]
        for label, bbox in zip(*target):
            st.append(str(label))
            st += [str(b) for b in bbox]
        return "\n" + " ".join(st)
    
    def load(self, base_directory):
        base_path = os.path.abspath(base_directory)
        if not os.path.exists(base_path) or (not os.path.isdir(base_path)):
            raise ValueError(f"{base_path} is not existed or is not a directory")
        image_sources = list()
        label_file = os.path.join(base_path, 'label.txt')
        with open(label_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
        for line in lines:
            try:
                items = [item.strip() for item in  line.split(" ")]
                image_path, *ls = items
                gts, bnds = list(), list()
                n = len(ls) // 5
                for i in range(n):
                    gts.append(ls[i*5])
                    bnds.append((int(ls[i*5+1]), int(ls[i*5+2]),
                                 int(ls[i*5+3]), int(ls[i*5+4])))
                    
                if len(image_path) > 0 and len(gts) > 0:
                    image_source = ImageSource(path=image_path, ground_truth=gts, bounding_box=bnds)
                    image_sources.append(image_source)
            except Exception as e:
                print(f"fail to read line as '{line}': {e.args}.")
        self.image_sources = image_sources
    
    def _random_operation(self, image, target):
        gts, bnds = target
        for operation in self.operations:
            r = np.random.uniform(0, 1)
            if r < operation.probability:
                image, bnds = operation.perform_with_box(image, bnds)
                target = target
        return image, (gts, bnds) 
    
        
            
class PoseImageSet(LabelImageSet):
    
    def scan_voc(self, base_directory, skip_difficult=True):
        if not (os.path.exists(base_directory) and os.path.isdir(base_directory)):
            raise IOError(f"there is no such directory named {base_directory}.")
        image_sources = list()
        annotation_directory = os.path.join(base_directory, _voc_annotation)
        jpg_directory = os.path.join(base_directory, _voc_images)
        
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
            pose_gts = list()
            pose_names = list()
            bnd_boxes = []
            index = 0
            for obj in root.iter("object"):
                if obj.find("pose") is None:
                    continue
                label = obj.find("name").text
                pose_name = obj.find("pose").text
                if pose_name == "Unspecified":
                    continue
                self._target.add(label)
                if obj.find("difficult") is None:
                    difficult = 0
                else:
                    difficult = int(obj.find("difficult").text)
                if skip_difficult and difficult == 1:
                    continue
                pose_gt = {"label": label}
                pose_xml = obj.find("bndbox")
                pose_names.append(pose_name)
                if pose_xml is not None:
                    pose_box = (int(pose_xml.find('xmin').text), int(pose_xml.find('ymin').text), 
                                int(pose_xml.find('xmax').text), int(pose_xml.find('ymax').text))
                    pose_gt["index"] = index
                    bnd_boxes.append(pose_box)
                    index += 1
                pose_parts = list()
                for part in obj.iter("part"):
                    part_label = part.find("name").text
                    box_xml = part.find("bndbox")
                    part_box = (int(box_xml.find('xmin').text), int(box_xml.find('ymin').text), 
                                int(box_xml.find('xmax').text), int(box_xml.find('ymax').text))
                    pose_parts.append({"label": part_label, "index": index})
                    bnd_boxes.append(part_box)
                    index += 1
                pose_gt["part"] = pose_parts
                pose_gts.append(pose_gt)
            if len(pose_gts) > 0:
                image_sources.append(ImageSource(path=image_path, pose=pose_names,
                                                 ground_truth=pose_gts, bounding_box=bnd_boxes))     
        self.image_sources =  image_sources
        
    
    def get_target(self, image_source):
        pose = image_source.pose
        gt = image_source.ground_truth
        bnds = image_source.bounding_box
        return pose, gt, bnds
    
    def dump_target(self, target, save_name):
        st = [save_name]
        pose, gts, bnds = target
        st.append("/".join([str(p) for p in pose]))
        for gt in gts:
            s = [str(gt["label"])]
            bb = bnds[gt["index"]]
            s += [str(b) for b in bb]
            for part in gt["part"]:
                s.append(str(part["label"]))
                pbb = bnds[part["index"]]
                s += [str(b) for b in pbb]
            st.append("/".join(s))
        return "\n" + " ".join(st)
    
    def load(self, base_directory):
        base_path = os.path.abspath(base_directory)
        if not os.path.exists(base_path) or (not os.path.isdir(base_path)):
            raise ValueError(f"{base_path} is not existed or is not a directory")
        image_sources = list()
        label_file = os.path.join(base_path, 'label.txt')
        with open(label_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
        for line in lines:
            try:
                items = [item.strip() for item in  line.split(" ")]
                image_path, pose_str, *ls = items
                pose_names = pose_str.split("/")
                gts, bnds = list(), list()
                index = 0
                for l in ls:
                    litems = l.split("/")
                    n = len(litems) // 5
                    pose_gt = {"part": []}
                    for i in range(n):
                        label = litems[i*5]
                        bnds.append((int(litems[i*5+1]), int(litems[i*5+2]),
                                     int(litems[i*5+3]), int(litems[i*5+4])))
                        if i == 0:
                            pose_gt["label"] = label
                            pose_gt["index"] = index
                        else:
                            pose_gt["part"].append({"label": label, "index": index})
                        index += 1
                    gts.append(pose_gt)
                if len(image_path) > 0 and len(gts) > 0:
                    image_source = ImageSource(path=image_path, pose=pose_names, ground_truth=gts, bounding_box=bnds)
                    image_sources.append(image_source)
            except Exception as e:
                print(f"fail to read line as '{line}': {e.args}.")
        self.image_sources = image_sources
    
    def _random_operation(self, image, target):
        pose, gts, bnds = target
        for operation in self.operations:
            r = np.random.uniform(0, 1)
            if r < operation.probability:
                image, bnds = operation.perform_with_box(image, bnds)
                target = target
        return image, (pose, gts, bnds) 
    
    
class SegmentClassImageSet(LabelImageSet):
    
    _segment_name = "segment_class"
    _segment_path =_voc_segclass
    
    
    def scan_voc(self, base_directory):
        if not (os.path.exists(base_directory) and os.path.isdir(base_directory)):
            raise IOError(f"there is no such directory named {base_directory}.")
        image_sources = list()
        annotation_directory = os.path.join(base_directory, _voc_annotation)
        jpg_directory = os.path.join(base_directory, _voc_images)
        segment_path = os.path.join(base_directory, self._segment_path)
        
        for xml_file in glob.glob(os.path.join(annotation_directory, "*.xml")):
            try:
                with open(xml_file, encoding='utf-8') as file:
                    tree = ET.parse(file)
            except IOError as e:
                print(f"can't read the file {xml_file}: {e.message}.")
                continue
            root = tree.getroot()
            image_file = root.find("filename").text
            base_file_name = os.path.splitext(image_file)[0]
            image_path = os.path.join(jpg_directory, image_file)
            segemented = root.find("segmented")
            if segemented is None or int(segemented.text) == 0:
                continue
            segment_image_path = os.path.join(segment_path, base_file_name + ".png")
            image_source = ImageSource(path=image_path)
            image_source.__setattr__(self._segment_name, segment_image_path)
            image_sources.append(image_source)
        self.image_sources =  image_sources
    
    
    def check_output_directory(self):
        if self.output_directory is None:
            raise ValueError("no output direcotry to save iamge, use 'set_output_directory' firstly.")
        if not (os.path.exists(self.output_directory) and os.path.isdir(self.output_directory)):
            try:
                os.mkdir(self.output_directory)
            except Exception:
                raise IOError("no authority to make a output directory.")
        self.output_target_path = os.path.join(self.output_directory, self._segment_path)
        if  not (os.path.exists(self.output_target_path) and os.path.isdir(self.output_directory)):
            try:
                os.mkdir(self.output_target_path)
            except Exception:
                raise IOError("no authority to make a output directory.")
    
    
    def get_target(self, image_source):
        return Image.open(image_source.__getattribute__(self._segment_name))
    
    def save_image(self, image, target):
        file_name = str(uuid.uuid4())
        try:
            save_name = f"{file_name}.{self.save_format}"
            image.save(os.path.join(self.output_directory, save_name))
            segment_out_path = os.path.join(self.output_directory, self._segment_path)
            target.save(os.path.join(segment_out_path, f"{file_name}.png"))
        except IOError as e:
            print("Error writing %s, %s. Change save_format to JPG?" % (file_name, e.message))
            print("You can change the save format using the set_save_format(save_format) function.")
            print("By passing save_format=\"auto\", Augmentor can save in the correct format automatically.")
    
    def dump_target(self, target, save_name):
        return ""
    
        
    def load(self, base_directory):
        base_path = os.path.abspath(base_directory)
        if not os.path.exists(base_path) or (not os.path.isdir(base_path)):
            raise ValueError(f"{base_path} is not existed or is not a directory")
        target_path = os.path.join(base_path, self._segment_path)
        image_sources = list()
        image_files = scan_images(base_path)
        for file in image_files:
            base_name = os.path.splitext(os.path.basename(file))[0]
            target_file = os.path.join(target_path, f"{base_name}.png")
            if os.path.exists(target_file):
                image_source = ImageSource(path=os.path.join(base_path, file))
                image_source.__setattr__(self._segment_name, target_file)
                image_sources.append(image_source)
        self.image_sources = image_sources
        
    def _random_operation(self, image, target):
        for operation in self.operations:
            r = np.random.uniform(0, 1)
            if r < operation.probability:
                image, target = operation.perform_with_segment(image, target)
                target = target
        return image, target

    

class SegmentObjectImageSet(SegmentClassImageSet):
     
    _segment_name = "segment_object"
    _segment_path = _voc_segobject
    