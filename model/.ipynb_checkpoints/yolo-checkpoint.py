from torch import nn
from torch import Tensor, flatten

class CNN(nn.Module):
    """
    
    """
    
    def __init__(self, in_channel:int, out_channel:int, kernel_size:int, stride=1, padding=0, bias=False):
        super(CNN,self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, tensor:Tensor)->Tensor:
        block = self.act(self.bn(self.conv(tensor)))
        return block

class FullyConnected(nn.Module):

    def __init__(self, split_size:int, num_boxes:int, num_classes:int):
        super(FullyConnected,self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*split_size*split_size, 4096),
            nn.Dropout(),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, (num_boxes*5+num_classes)*split_size*split_size)
        )
    def forward(self, tensor: Tensor):
        tensor = flatten(tensor, start_dim=1)
        fc = self.fc(tensor)
        return fc

class Darknet(nn.Module):
    
    def __init__(self):
        super(Darknet, self).__init__()
        self.main = nn.Sequential(
            CNN(3, 64, 7, 2, 3),
            nn.MaxPool2d(2,2),
            CNN(64,192,1,1),
            nn.MaxPool2d(2,2),
            CNN(192,128,1),
            CNN(128,256,3,1,1),
            CNN(256,256,1),
            CNN(256,512,3,1,1),
            nn.MaxPool2d(2,2),
            CNN(512,256,1,1,0),CNN(256,512,3,1,1),
            CNN(512,256,1,1,0),CNN(256,512,3,1,1),
            CNN(512,256,1,1,0),CNN(256,512,3,1,1),
            CNN(512,256,1,1,0),CNN(256,512,3,1,1),
            CNN(512,512,1),
            CNN(512,1024,3,1,1),
            nn.MaxPool2d(2,2),
            CNN(1024,512,1,1,0),CNN(512,1024,3,1,1),
            CNN(1024,512,1,1,0),CNN(512,1024,3,1,1),
            CNN(1024,1024,3,1,1),
            CNN(1024,1024,3,2,1), CNN(1024,1024,3,1,1), CNN(1024,1024,3,1,1)
        )

    def forward(self, tensor:Tensor):
        darknet = self.main(tensor)
        return darknet

class Yolo(nn.Module):
    
    def __init__(self, split_size:int, num_boxes:int, num_classes:int):
        super(Yolo, self).__init__()
        self.darknet = Darknet()
        self.fc = FullyConnected(split_size, num_boxes, num_classes)
    
    def forward(self, tensor:Tensor):
        return self.fc(self.darknet(tensor))