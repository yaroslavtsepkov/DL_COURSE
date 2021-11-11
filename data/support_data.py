from torchvision.transforms.functional import adjust_sharpness
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms as T
import torch
import pandas as pd
import os

def show_bbox(img:torch.Tensor, bbox:torch.Tensor, **kwargs):
    bbox = converter(bbox)
    tensor_bbox = draw_bounding_boxes(img, bbox, **kwargs)
    img_bbox = T.ToPILImage()(tensor_bbox)
    return img_bbox

def converter(bbox: torch.Tensor)->torch.Tensor:
    """
    Chagne format bbox: from x, y, w, h -> xmin, ymin, xmax, ymax
    Args:
        bbox(Tensor): tensor with bounded boxes.
    Return:
        bbox(Tensor): new format bounded boxes.
    """

    dims, coords = bbox.size()
    assert coords == 4, "Error boxes format"
    assert dims > 0, "Error boxes format"
    new_bbox = torch.empty_like(bbox)
    new_bbox[:, :2] = bbox[:, :2]
    new_bbox[:, 2:] = bbox[:, :2] + bbox[:, 2:]
    return new_bbox

def createCSV(split="train", output="data/dataset"):
    annot = {
        "filename":[],
        "bboxes":[]
    }
    with open(f"/home/essea/repos/Yolo/data/wider_face_split/wider_face_{split}_bbx_gt.txt","r") as f:
        while True:
            name = f.readline().rstrip()
            if not name:
                break

            try:
                numfaces = int(f.readline().rstrip())
            except:
                continue
            temp = []
            for _ in range(numfaces):
                row = list(map(int, f.readline().rstrip().split()[:4]))
                temp.append(row)
            annot['filename'].append(name)
            annot['bboxes'].append(temp)
    df = pd.DataFrame.from_dict(annot)
    df["filename"] = df["filename"].map(lambda x: os.path.join(f"/home/essea/datasets/WIDER_{split}/",x))
    output = os.path.join(output, f"{split}.csv")
    df.to_csv(output)

if __name__ == '__main__':
    pass