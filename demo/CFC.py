import torch
import torch.nn as nn
import torch.nn.functional as F
class CFC_Module(nn.Module):
    def __init__(self):
        super(CFC_Module, self).__init__()
        self.uplayer3 = nn.Sequential(
            nn.Conv2d(64, 16, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.uplayer4 = nn.Sequential(
            nn.Conv2d(32, 16, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.alpha1 = torch.nn.Parameter(torch.randn(1))
        self.alpha2 = torch.nn.Parameter(torch.randn(1))
        self.alpha3 = torch.nn.Parameter(torch.randn(1))
        self.conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, low, mid, high):
        high = self.uplayer3(high)
        high2low = F.interpolate(high, scale_factor=4, mode='nearest')
        mid = self.uplayer4(mid)
        mid2low = F.interpolate(mid, scale_factor=2, mode='nearest')
        out1 = self.alpha1 * low + self.alpha2 * high2low + self.alpha3 * mid2low
        out1 = out1 + low
        avg_filter = torch.mean(high, dim=1, keepdim=True)
        max_filter, _ = torch.max(high, dim=1, keepdim=True)
        kernel_filter = torch.cat([avg_filter, max_filter], dim=1)
        kernel_filter = self.sigmoid(self.conv(kernel_filter))
        kernel_filter = F.interpolate(kernel_filter, scale_factor=2, mode='nearest')
        Purity = mid * kernel_filter
        Purity = F.interpolate(Purity, scale_factor=2, mode='nearest')
        HSNR = Purity + out1
        
        return HSNR