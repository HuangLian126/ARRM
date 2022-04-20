import os
import torch
import random
import collections
import torch.utils.data
from PIL import Image
import numpy as np
import sys
import math
from maskrcnn_benchmark.data.transforms import transforms as T
import random
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from maskrcnn_benchmark.structures.bounding_box import BoxList

class PascalVOCDatasetSUP(torch.utils.data.Dataset):
    CLASSES = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )
    CLASSES_SPLIT1_BASE = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "boat",
        "bottle",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
    )
    CLASSES_SPLIT1_BASEXX = (
        "aeroplane",
        "bicycle",
        "boat",
        "bottle",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
    )

    CLASSES_SPLIT2_BASE = (
        "__background__ ",
        "bicycle",
        "bird",
        "boat",
        "bus",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
    )
    CLASSES_SPLIT3_BASE = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "bottle",
        "bus",
        "car",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "train",
        "tvmonitor",
    )
    CLASSES_SPLIT1_NOVEL = (
        "bird",
        "bus",
        "cow",
        "motorbike",
        "sofa",
    )
    CLASSES_SPLIT2_NOVEL = (
        "aeroplane",
        "bottle",
        "cow",
        "horse",
        "sofa"
    )
    CLASSES_SPLIT3_NOVEL = (
        "boat",
        "cat",
        "motorbike",
        "sheep",
        "sofa",
    )

    def __init__(self, data_dir, split, use_difficult=False, transforms=None, toofew=True):
        self.root           = "/home/hl/hl/MPSR-master/" + data_dir  # 根目录路径
        self.image_set      = split
        self.keep_difficult = use_difficult
        self.transforms     = transforms
        self._annopath      = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath       = os.path.join(self.root, "JPEGImages",  "%s.jpg")
        self._imgsetpath    = os.path.join(self.root, "ImageSets",   "Main", "%s.txt")

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        # print('self.image_set: ', self.image_set)
        # print('self.ids: ', self.ids)

        # too few ids lead to an unfixed bug in dataloader
        if len(self.ids) < 50 and toofew:
            self.ids = self.ids * (int(100 / len(self.ids)) + 1)

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}  # {0:22, 1:256, 2:963, .......}
        # print('self.id_to_img_map: ', self.id_to_img_map)

        if 'split1_base' in split:
            cls = PascalVOCDatasetSUP.CLASSES_SPLIT1_BASE
        elif 'split2_base' in split:
            cls = PascalVOCDatasetSUP.CLASSES_SPLIT2_BASE
        elif 'split3_base' in split:
            cls = PascalVOCDatasetSUP.CLASSES_SPLIT3_BASE
        else:
            cls = PascalVOCDatasetSUP.CLASSES
        self.cls = cls

        self.clsXX = PascalVOCDatasetSUP.CLASSES_SPLIT1_BASEXX

        self.class_to_ind = dict(zip(cls, range(len(cls))))  # {'__background__ ': 0, 'aeroplane': 1, 'bicycle': 2, 'boat': 3, 'bottle': 4, 'car': 5, 'cat': 6, 'chair': 7, 'diningtable': 8, 'dog': 9, 'horse': 10, 'person': 11, 'pottedplant': 12, 'sheep': 13, 'train': 14, 'tvmonitor': 15}

        self.categories = dict(zip(range(len(cls)), cls))
        # print('self.categories: ', self.categories)  #{0: '__background__ ', 1: 'aeroplane', 2: 'bicycle', 3: 'boat', 4: 'bottle', 5: 'car', 6: 'cat', 7: 'chair', 8: 'diningtable', 9: 'dog', 10: 'horse', 11: 'person', 12: 'pottedplant', 13: 'sheep', 14: 'train', 15: 'tvmonitor'}


        self.prn_image = collections.defaultdict(list)
        # print('prn_image: ', prn_image)
        self.classes = collections.defaultdict(int)
        for clss in self.clsXX:
            self.classes[clss] = 0
        # print('classes: ', classes)   # {'aeroplane': 0, 'bicycle': 0, 'boat': 0, 'bottle': 0, 'car': 0, 'cat': 0, 'chair': 0, 'diningtable': 0, 'dog': 0, 'horse': 0, 'person': 0, 'pottedplant': 0, 'sheep': 0, 'train': 0, 'tvmonitor': 0})

        for index_id1 in range(0, len(self.ids)):
            img1 = Image.open(self._imgpath % self.ids[index_id1]).convert("RGB")
            img1 = img1.resize((224, 244))

            target1 = self.get_groundtruth(index_id1)
            target1 = target1.clip_to_image(remove_empty=True)
            if self.transforms is not None:
                img1, target1 = self.transforms(img1, target1)
            labelS = target1.get_field("labels").numpy()
            # print('labelS: ', labelS)   # cls是训练数据所包含的类别
            for i in range(0, len(labelS)):
                name = self.cls[labelS[i]]
                gtMask1 = torch.zeros(1, img1.shape[1], img1.shape[2])

                bbox_i = target1.bbox[i]
                if self.classes[name] >= 200:
                    break
                self.classes[name] += 1
                gtMask1[int(torch.ceil(bbox_i[1])):int(torch.ceil(bbox_i[3])), int(torch.floor(bbox_i[0])):int(torch.floor(bbox_i[2]))] = 255
                imgMaskgg1 = torch.cat((img1, gtMask1), dim=0)
                self.prn_image[name].append(imgMaskgg1)
            if len(self.classes) > 0 and min(self.classes.values()) == 200:
                break
        # print(min(self.classes.values()))

    def __getitem__(self, index):

        datax = []
        for n, key in enumerate(list(self.prn_image.keys())):
            imgfs = self.prn_image[key][index]
            datax.append(imgfs)
        imgMask = torch.cat(datax, dim=0)
        # print('imgMask.shape: ', imgMask.shape)
        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)
        return imgMask, target, index

        '''
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")
        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        print('target', target.get_field("labels"))

        bndbox = target.bbox[0].numpy()
        gtMask = torch.zeros(1, target.size[1], target.size[0])
        gtMask[int(bndbox[1]):int(bndbox[3]), int(bndbox[0]):int(bndbox[2])] = 255
        imgMask = torch.cat((img, gtMask), dim=0)
        imgMaskgg  = torch.cat((img, gtMask), dim=0)
        imgMaskgg2 = torch.cat((img, gtMask), dim=0)
        imgMaskgg3 = torch.cat((img, gtMask), dim=0)
        imgMaskgg4 = torch.cat((img, gtMask), dim=0)
        imgMaskgg5 = torch.cat((img, gtMask), dim=0)
        imgMaskgg6 = torch.cat((img, gtMask), dim=0)
        imgMaskgg7 = torch.cat((img, gtMask), dim=0)
        imgMaskgg8 = torch.cat((img, gtMask), dim=0)
        imgMaskgg9 = torch.cat((img, gtMask), dim=0)
        imgMaskgg10 = torch.cat((img, gtMask), dim=0)
        imgMaskgg11 = torch.cat((img, gtMask), dim=0)
        imgMaskgg12 = torch.cat((img, gtMask), dim=0)
        imgMaskgg13 = torch.cat((img, gtMask), dim=0)
        imgMaskgg14 = torch.cat((img, gtMask), dim=0)
        imgMaskgg15 = torch.cat((img, gtMask), dim=0)

        imgMaskgg664 = torch.cat((imgMaskgg, imgMaskgg2, imgMaskgg3, imgMaskgg4, imgMaskgg5, imgMaskgg6, imgMaskgg7, imgMaskgg8, imgMaskgg9, imgMaskgg10, imgMaskgg11, imgMaskgg12, imgMaskgg13, imgMaskgg14, imgMaskgg15), dim=0)
        # print('imgMaskgg664: ', imgMaskgg664.shape)
        return imgMaskgg664, target, index
        '''


    def __len__(self):
        #return len(self.ids)
        return min(self.classes.values())

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)
        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1
        size = target.find("size")
        heightOri = int(size.find("height").text)
        widthOri  = int(size.find("width").text)
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [bb.find("xmin").text, bb.find("ymin").text, bb.find("xmax").text, bb.find("ymax").text]

            box[0] = math.ceil(int(box[0]) * 224 / widthOri)
            box[2] = math.ceil(int(box[2]) * 224 / widthOri)
            box[1] = math.floor(int(box[1]) * 224 / heightOri)
            box[3] = math.floor(int(box[3]) * 224 / heightOri)

            bndbox = tuple(map(lambda x: x - TO_REMOVE, list(map(int, box))))

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)
        res = {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(gt_classes), "difficult": torch.tensor(difficult_boxes), "im_info": im_info}
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        # return PascalVOCDataset.CLASSES[class_id]
        return self.cls[class_id]