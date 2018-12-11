import torch
import torch.utils.data as data
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np

from utils.augmentations import Compose, ConvertFromInts, ToAbsoluteCoords, ToPercentCoords, Resize, SubtractMeans


class IDDDataset(data.Dataset):
    def __init__(self, root: str, image_folder: str, annos_folder: str, size: int = 300):
        """

        :param root: root path to dataset folder
        :param image_folder: relative path to image folder
        :param annos_folder: relative path to annos folder
        :param transforms: NONE for now
        """
        self.mean = (104, 117, 123)  # TODO: figure out why
        self.root = Path(root)
        self.image_folder = self.root / image_folder
        self.annos_folder = self.root / annos_folder
        self.fns = os.listdir(self.image_folder)
        self.transforms = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(self.mean)
        ])
        # self.fns_wo_ext = [os.path.splitext(os.path.basename(fn))[0] for fn in self.fns]
        self.id2fn = {idx: fn for idx, fn in enumerate(self.fns)}
        self.fn2id = {fn: idx for idx, fn in self.id2fn.items()}

    def __getitem__(self, index):
        fn = self.id2fn[index]
        img_fn = self.image_folder / fn
        anno_fn = self.annos_folder / (os.path.splitext(fn)[0] + '.xml')
        img = cv2.imread(str(img_fn))
        h, w, c = img.shape
        anno = self.read_anno(anno_fn)
        img, boxes, labels = self.transforms(img, anno[:, :4], anno[:, 4])
        img = img[:, :, (2, 1, 0)]
        # img = img.transpose(2, 0, 1)
        target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # im = torch.from_numpy(im).permute(2, 0, 1)
        return img, target

    def read_anno(self, f):
        """

        :param f: full path to the xml file
        :return:
        """
        t = ET.parse(f)
        r = t.getroot()

        # folder = r.findall('folder')[0].text
        filename = r.findall('filename')[0].text
        xmin = int(r.findall('object/bndbox/xmin')[0].text)
        ymin = int(r.findall('object/bndbox/ymin')[0].text)
        xmax = int(r.findall('object/bndbox/xmax')[0].text)
        ymax = int(r.findall('object/bndbox/ymax')[0].text)
        clas = 0
        return np.array([[xmin, ymin, xmax, ymax, clas]])
        # clas = 'ID'
        # return {
        #     'id': idx,
        #     'filename': filename,
        #     'bbox': f'{xmin} {ymin} {xmax} {ymax}',
        #     'clas': clas
        # }

    def __len__(self):
        return len(self.fns)
