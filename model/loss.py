import torch
import torch.nn as nn
import torch.nn.functional as F

def trans_rot2Tmat(trans1_2, rot1_2):
    bs = trans1_2.shape[0]
    Tmat = torch.zeros((bs, 4, 4))
    
    return Tmat

def Warp(image1, Tmat):
    mask = torch.zeros_like(image1)
    
    return image1, mask


if __name__ == "__main__":
    input = torch.randn((4, 3, 960, 540))       # Scale = (0.5, 0.5)
    input = F.pad(input, (2, 2, 0, 0), "constant", 0)
    target = torch.randn((4, 3, 960, 540))       # Scale = (0.5, 0.5)
    target = F.pad(target, (2, 2, 0, 0), "constant", 0)
    
    trans = torch.rand((4, 3))
    rot = torch.rand((4, 3))
    
    Tmat = trans_rot2Tmat(trans, rot)
    
    input, mask = Warp(input, Tmat)
    target = target * mask
    
    smothl1loss = nn.SmoothL1Loss()
    loss = smothl1loss(input, target)
    
    print(loss)