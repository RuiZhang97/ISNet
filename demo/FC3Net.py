import torch
import torch.nn as nn
import torch.nn.functional as F
from FCNHead import _FCNHead
from Basic import FC3BasicBlock
from CFC import CFC_Module
BN_MOMENTUM = 0.2

class FC3(nn.Module):
    def __init__(self, channels):
        super(FC3, self).__init__()
        
        self.input_bn0 = nn.BatchNorm2d(3, momentum=BN_MOMENTUM)
        self.stem1 = FC3BasicBlock(inchannels=3, outchannels=32, stride=2, padding=1)
        self.stem2 = FC3BasicBlock(inchannels=32, outchannels=32, stride=1, padding=1)

        self.input = nn.Conv2d(32, 16, kernel_size=1, stride=1, bias=False)
        self.input_bn = nn.BatchNorm2d(16, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()

        self.lay1 = FC3BasicBlock(inchannels=16, outchannels=32, stride=2, padding=1)
        self.lay2 = FC3BasicBlock(inchannels=32, outchannels=64, stride=2, padding=1)
        self.lay3 = FC3BasicBlock(inchannels=64, outchannels=128, stride=2, padding=1)

        self.resb1 = FC3BasicBlock(inchannels=16, outchannels=16, stride=1, padding=1)
        self.resb2 = FC3BasicBlock(inchannels=16, outchannels=16, stride=1, padding=1)
        self.resb3 = FC3BasicBlock(inchannels=16, outchannels=16, stride=1, padding=1)
        self.resb4 = FC3BasicBlock(inchannels=16, outchannels=16, stride=1, padding=1)
        self.resb5 = FC3BasicBlock(inchannels=16, outchannels=16, stride=1, padding=1)
        self.resb6 = FC3BasicBlock(inchannels=16, outchannels=16, stride=1, padding=1)
        self.b3_down = FC3BasicBlock(inchannels=16, outchannels=32, stride=2, padding=1)
        self.b5_down = FC3BasicBlock(inchannels=16, outchannels=32, stride=2, padding=1)
        self.b7_down = FC3BasicBlock(inchannels=16, outchannels=32, stride=2, padding=1)


        self.resc1_1 = FC3BasicBlock(inchannels=32, outchannels=32, stride=1, padding=1)
        self.resc1_2 = FC3BasicBlock(inchannels=32, outchannels=32, stride=1, padding=1)
        self.resc1_3 = FC3BasicBlock(inchannels=32, outchannels=32, stride=1, padding=1)
        self.resc1_4 = FC3BasicBlock(inchannels=32, outchannels=32, stride=1, padding=1)
        self.resc1_5 = FC3BasicBlock(inchannels=32, outchannels=32, stride=1, padding=1)
        self.resc1_6 = FC3BasicBlock(inchannels=32, outchannels=32, stride=1, padding=1)
        self.c1_down = FC3BasicBlock(inchannels=32, outchannels=64, stride=2, padding=1)

        self.resc1_d_1 = FC3BasicBlock(inchannels=64, outchannels=64, stride=1, padding=1)
        self.resc1_d_2 = FC3BasicBlock(inchannels=64, outchannels=64, stride=1, padding=1)
        self.resc1_d_3 = FC3BasicBlock(inchannels=64, outchannels=64, stride=1, padding=1)
        self.resc1_d_4 = FC3BasicBlock(inchannels=64, outchannels=64, stride=1, padding=1)
        self.resc1_d_5 = FC3BasicBlock(inchannels=64, outchannels=64, stride=1, padding=1)
        self.resc1_d_6 = FC3BasicBlock(inchannels=64, outchannels=64, stride=1, padding=1)
        self.c1_d_3_up =  nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU()
        )
        self.c1_d_5_up = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU()
        )
        self.c1_d_7_up = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU()
        )

        self.resc1c2_1 = FC3BasicBlock(inchannels=32, outchannels=32, stride=1, padding=1)
        self.resc1c2_2 = FC3BasicBlock(inchannels=32, outchannels=32, stride=1, padding=1)
        self.resc1c2_3 = FC3BasicBlock(inchannels=32, outchannels=32, stride=1, padding=1)
        self.resc1c2_4 = FC3BasicBlock(inchannels=32, outchannels=32, stride=1, padding=1)
        self.resc1c2_5 = FC3BasicBlock(inchannels=32, outchannels=32, stride=1, padding=1)
        self.resc1c2_6 = FC3BasicBlock(inchannels=32, outchannels=32, stride=1, padding=1)
        self.c1c2_3_down = FC3BasicBlock(inchannels=32, outchannels=64, stride=2, padding=1)
        self.c1c2_5_down = FC3BasicBlock(inchannels=32, outchannels=64, stride=2, padding=1)
        self.c1c2_7_down = FC3BasicBlock(inchannels=32, outchannels=64, stride=2, padding=1)


        self.resc2_1 = FC3BasicBlock(inchannels=64, outchannels=64, stride=1, padding=1)
        self.resc2_2 = FC3BasicBlock(inchannels=64, outchannels=64, stride=1, padding=1)
        self.resc2_3 = FC3BasicBlock(inchannels=64, outchannels=64, stride=1, padding=1)
        self.resc2_4 = FC3BasicBlock(inchannels=64, outchannels=64, stride=1, padding=1)
        self.resc2_5 = FC3BasicBlock(inchannels=64, outchannels=64, stride=1, padding=1)
        self.resc2_6 = FC3BasicBlock(inchannels=64, outchannels=64, stride=1, padding=1)
        self.resc2_d = FC3BasicBlock(inchannels=64, outchannels=128, stride=2, padding=1)

        self.resc2_d_1 = FC3BasicBlock(inchannels=128, outchannels=128, stride=1, padding=1)
        self.resc2_d_2 = FC3BasicBlock(inchannels=128, outchannels=128, stride=1, padding=1)
        self.resc2_d_3 = FC3BasicBlock(inchannels=128, outchannels=128, stride=1, padding=1)
        self.resc2_d_4 = FC3BasicBlock(inchannels=128, outchannels=128, stride=1, padding=1)
        self.resc2_d_5 = FC3BasicBlock(inchannels=128, outchannels=128, stride=1, padding=1)
        self.resc2_d_6 = FC3BasicBlock(inchannels=128, outchannels=128, stride=1, padding=1)
        self.c2_d_3_up = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU()
        )
        self.c2_d_5_up = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU()
        )
        self.c2_d_7_up = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU()
        )

        self.resc2c3_1 = FC3BasicBlock(inchannels=64, outchannels=64, stride=1, padding=1)
        self.resc2c3_2 = FC3BasicBlock(inchannels=64, outchannels=64, stride=1, padding=1)
        self.resc2c3_3 = FC3BasicBlock(inchannels=64, outchannels=64, stride=1, padding=1)
        self.resc2c3_4 = FC3BasicBlock(inchannels=64, outchannels=64, stride=1, padding=1)
        self.resc2c3_5 = FC3BasicBlock(inchannels=64, outchannels=64, stride=1, padding=1)
        self.resc2c3_6 = FC3BasicBlock(inchannels=64, outchannels=64, stride=1, padding=1)
        self.c2c3_3_down = FC3BasicBlock(inchannels=64, outchannels=128, stride=2, padding=1)
        self.c2c3_5_down = FC3BasicBlock(inchannels=64, outchannels=128, stride=2, padding=1)
        self.c2c3_7_down = FC3BasicBlock(inchannels=64, outchannels=128, stride=2, padding=1)

        self.resc3_1 = FC3BasicBlock(inchannels=128, outchannels=128, stride=1, padding=1)
        self.resc3_2 = FC3BasicBlock(inchannels=128, outchannels=128, stride=1, padding=1)
        self.resc3_3 = FC3BasicBlock(inchannels=128, outchannels=128, stride=1, padding=1)
        self.resc3_4 = FC3BasicBlock(inchannels=128, outchannels=128, stride=1, padding=1)
        self.resc3_5 = FC3BasicBlock(inchannels=128, outchannels=128, stride=1, padding=1)
        self.resc3_6 = FC3BasicBlock(inchannels=128, outchannels=128, stride=1, padding=1)

        self.resc3_d = FC3BasicBlock(inchannels=128, outchannels=256, stride=2, padding=1)
        self.resc3_d_1 = FC3BasicBlock(inchannels=256, outchannels=256, stride=1, padding=1)
        self.resc3_d_2 = FC3BasicBlock(inchannels=256, outchannels=256, stride=1, padding=1)
        self.resc3_d_3 = FC3BasicBlock(inchannels=256, outchannels=256, stride=1, padding=1)
        self.resc3_d_4 = FC3BasicBlock(inchannels=256, outchannels=256, stride=1, padding=1)
        self.resc3_d_5 = FC3BasicBlock(inchannels=256, outchannels=256, stride=1, padding=1)
        self.resc3_d_6 = FC3BasicBlock(inchannels=256, outchannels=256, stride=1, padding=1)
        self.c3_d_3_up = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU()
        )
        self.c3_d_5_up = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU()
        )
        self.c3_d_7_up = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU()
        )
        self.c3_up = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU()
        )
        self.c2_up = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU()
        )
        self.c1_up = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1),
            nn.BatchNorm2d(16, momentum=BN_MOMENTUM),
            nn.ReLU()
        )
        
        self.CFC = CFC_Module()

        self.head = _FCNHead(channels[1], 1)


    def forward(self, x):
        _, _, hei, wid = x.shape

        x_bn = self.input_bn0(x)
        B1 = self.stem1(x_bn)
        B1 = self.stem2(B1)
        B1 = self.input(B1)
        B1 = self.input_bn(B1)
        B1 = self.relu(B1)
        
        c1 = self.lay1(B1)
        c2 = self.lay2(c1)
        c3 = self.lay3(c2)
        
        
        #The first branch of Multi-parallel MFC#
        B2 = self.resb1(B1)
        B3 = self.resb2(B2)
        B4 = self.resb3(B3)
        B5 = self.resb4(B4)
        B6 = self.resb5(B5)
        B7 = self.resb6(B6)

        c1_d_1 = self.c1_down(c1)
        c1_d_2 = self.resc1_d_1(c1_d_1)
        c1_d_3 = self.resc1_d_2(c1_d_2)
        c1_d_4 = self.resc1_d_3(c1_d_3)
        c1_d_5 = self.resc1_d_4(c1_d_4)
        c1_d_6 = self.resc1_d_5(c1_d_5)
        c1_d_7 = self.resc1_d_6(c1_d_6)

        c1_2 = self.resc1_1(c1)
        c1_3 = self.resc1_2(c1_2)

        B3_down = self.b3_down(B3)
        c1_d_3_up = F.interpolate(c1_d_3, scale_factor=2, mode='bilinear')
        c1_d_3_up = self.c1_d_3_up(c1_d_3_up)

        c1_3_add = c1_3 + B3_down + c1_d_3_up
        c1_4 = self.resc1_3(c1_3_add)
        c1_5 = self.resc1_4(c1_4)
        B5_down = self.b5_down(B5)
        c1_d_5_up = F.interpolate(c1_d_5, scale_factor=2, mode='bilinear')
        c1_d_5_up = self.c1_d_5_up(c1_d_5_up)
        c1_5_add = c1_5 + B5_down + c1_d_5_up

        c1_6 = self.resc1_5(c1_5_add)
        c1_7 = self.resc1_6(c1_6)

        B7_down = self.b7_down(B7)
        c1_d_7_up = F.interpolate(c1_d_7, scale_factor=2, mode='bilinear')
        c1_d_7_up = self.c1_d_7_up(c1_d_7_up)

        c1_7_add = c1_7 + B7_down + c1_d_7_up


        #The second branch of Multi-parallel MFC#
        c1c2_2 = self.resc1c2_1(c1)
        c1c2_3 = self.resc1c2_2(c1c2_2)
        c1c2_4 = self.resc1c2_3(c1c2_3)
        c1c2_5 = self.resc1c2_4(c1c2_4)
        c1c2_6 = self.resc1c2_5(c1c2_5)
        c1c2_7 = self.resc1c2_6(c1c2_6)

        c2_d_1 = self.resc2_d(c2)
        c2_d_2 = self.resc2_d_1(c2_d_1)
        c2_d_3 = self.resc2_d_2(c2_d_2)
        c2_d_4 = self.resc2_d_3(c2_d_3)
        c2_d_5 = self.resc2_d_4(c2_d_4)
        c2_d_6 = self.resc2_d_5(c2_d_5)
        c2_d_7 = self.resc2_d_6(c2_d_6)

        c2_2 = self.resc2_1(c2)
        c2_3 = self.resc2_2(c2_2)
        c1c2_3_down = self.c1c2_3_down(c1c2_3)
        c2_d_3_up = F.interpolate(c2_d_3, scale_factor=2, mode='bilinear')
        c2_d_3_up = self.c2_d_3_up(c2_d_3_up)
        c2_3_add = c2_3 + c1c2_3_down + c2_d_3_up
        c2_4 = self.resc2_3(c2_3_add)
        c2_5 = self.resc2_4(c2_4)
        c1c2_5_down = self.c1c2_5_down(c1c2_5)
        c2_d_5_up = F.interpolate(c2_d_5, scale_factor=2, mode='bilinear')
        c2_d_5_up = self.c2_d_5_up(c2_d_5_up)
        c2_5_add = c2_5 + c1c2_5_down + c2_d_5_up
        c2_6 = self.resc2_5(c2_5_add)
        c2_7 = self.resc2_6(c2_6)
        c1c2_7_down = self.c1c2_7_down(c1c2_7)
        c2_d_7_up = F.interpolate(c2_d_7, scale_factor=2, mode='bilinear')
        c2_d_7_up = self.c2_d_7_up(c2_d_7_up)
        c2_7_add = c2_7 + c1c2_7_down + c2_d_7_up

        
        #The third branch of Multi-parallel MFC#
        c2c3_2 = self.resc2c3_1(c2)
        c2c3_3 = self.resc2c3_2(c2c3_2)
        c2c3_4 = self.resc2c3_3(c2c3_3)
        c2c3_5 = self.resc2c3_4(c2c3_4)
        c2c3_6 = self.resc2c3_5(c2c3_5)
        c2c3_7 = self.resc2c3_6(c2c3_6)

        c3_d_1 = self.resc3_d(c3)
        c3_d_2 = self.resc3_d_1(c3_d_1)
        c3_d_3 = self.resc3_d_2(c3_d_2)
        c3_d_4 = self.resc3_d_3(c3_d_3)
        c3_d_5 = self.resc3_d_4(c3_d_4)
        c3_d_6 = self.resc3_d_5(c3_d_5)
        c3_d_7 = self.resc3_d_6(c3_d_6)

        c3_2 = self.resc3_1(c3)
        c3_3 = self.resc3_2(c3_2)
        c2c3_3_down = self.c2c3_3_down(c2c3_3)
        c3_d_3_up = F.interpolate(c3_d_3, scale_factor=2, mode='bilinear')
        c3_d_3_up = self.c3_d_3_up(c3_d_3_up)
        c3_3_add = c3_3 + c2c3_3_down + c3_d_3_up
        c3_4 = self.resc3_4(c3_3_add)
        c3_5 = self.resc3_5(c3_4)
        c2c3_5_down = self.c2c3_5_down(c2c3_5)
        c3_d_5_up = F.interpolate(c3_d_5, scale_factor=2, mode='bilinear')
        c3_d_5_up = self.c3_d_5_up(c3_d_5_up)
        c3_5_add = c3_5 + c2c3_5_down + c3_d_5_up
        c3_6 = self.resc3_5(c3_5_add)
        c3_7 = self.resc3_6(c3_6)
        c2c3_7_down = self.c2c3_7_down(c2c3_7)
        c3_d_7_up = F.interpolate(c3_d_7, scale_factor=2, mode='bilinear')
        c3_d_7_up = self.c3_d_7_up(c3_d_7_up)
        c3_7_add = c3_7 + c2c3_7_down + c3_d_7_up

        
        c3_7_add_up = F.interpolate(c3_7_add, scale_factor=2, mode='bilinear')
        c3_7_add_up = self.c3_up(c3_7_add_up)
        up1 = c3_7_add_up + c2_7_add
       
        up1_up = F.interpolate(up1, scale_factor=2, mode='bilinear')
        up1_up = self.c2_up(up1_up)
        
        up2 = up1_up + c1_7_add

        up2_up = F.interpolate(up2, scale_factor=2, mode='bilinear')
        up2_up = self.c1_up(up2_up)

        up3_1 = up2_up + B7


        #CFC#
        head_input = self.CFC(up3_1, up2, up1)
        pred = self.head(head_input)
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear')

        return out


