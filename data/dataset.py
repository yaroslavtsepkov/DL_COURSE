from torch.utils.data import Dataset
import pandas as pd
from typing import *
from PIL import Image
import re
import numpy as np
from utilities import show_bbox
import torch
from torchvision import transforms as T

class CWiderFace(Dataset):

    def __init__(self, csvfile: str, transform_img=None) -> None:
        """
        Args:
            csvfile(str): path to csv file (train, val, test)
            tranform_img(torch.Compose): transforms for images, like torch Compose
        """

        super(CWiderFace, self).__init__()
        self.files = pd.read_csv(csvfile)
        self.transform_img = transform_img

    def __len__(self):
        return self.files.shape[0]

    def __getitem__(self, index):
        filename, bbox = self.files.iloc[index]
        bbox = re.sub("\[+|\]+|\s+","",bbox).split(",")
        bbox = np.reshape(bbox, (len(bbox)//4, 4)).astype(int)
        bbox = torch.from_numpy(bbox)
        img = Image.open(filename)
        h, w = img.size
        if self.transform_img:
            img = self.transform_img(img)
            _, nh, nw = img.size()
            scale_x, scale_y = nh/h, nw/w
            bbox[:,0] = bbox[:,0]*scale_x
            bbox[:,1] = bbox[:,1]*scale_y
            bbox[:,2] = bbox[:,2]*scale_x
            bbox[:,3] = bbox[:,3]*scale_y
        return img, bbox

def main():
    tf_img = T.Compose([
        T.PILToTensor(),
        T.Resize((448,448))
    ])
    dataset = CWiderFace("/home/essea/repos/Yolo/data/dataset/train.csv",transform_img=tf_img)
    img, bbox = dataset[48]
    show_bbox(img, bbox, width=1).show()

if __name__ == '__main__':
    main()