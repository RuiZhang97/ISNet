import torch
from FC3Net import FC3


def test():
    FC3_Net = FC3(
        channels = [8, 16, 32, 64]
    )

    img = torch.randn(1, 3, 256, 256)
    
    preds = FC3_Net(img)
    print(preds.shape)
    
test()
