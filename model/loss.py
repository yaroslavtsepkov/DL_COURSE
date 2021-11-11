import torch
from torch import nn
from utilities import intersection_over_union as iou

class YoloLoss(nn.Module):

    def __init__(self, split:int, boxes:int, classes:int):
        """
        Args:        
            split(int): 
        """

        super(YoloLoss, self).__init__()
        self.split = split
        self.boxes = boxes
        self.classes = classes
        self.mse = nn.MSELoss(reduction="sum")
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        prediction = prediction.reshape(-1, self.split, self.split, self.classes + self.boxes * 5)

        # calc IOU
        io1 = iou(prediction, target)
        io2 = iou(prediction, target)
        ious = torch.cat([io1.unsqueeze(0), io2.unsqueeze(0)], dim=0)

        ioi_max, bestbox = torch.max(ious, dim=0)
        return prediction


