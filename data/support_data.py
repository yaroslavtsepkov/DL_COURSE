from torchvision.transforms.functional import adjust_sharpness
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms as T
import torch

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

if __name__ == '__main__':
    pass