import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

def trans_rot2Tmat(trans1_2, rot1_2):
    bs = trans1_2.shape[0]
    device = trans1_2.device
    
    cos_rot = torch.cos(rot1_2)
    sin_rot = torch.sin(rot1_2)
    
    batch_ones = torch.ones((bs), device = device)
    batch_zeros = torch.zeros((bs), device = device)
    
    col0 = torch.stack((batch_ones, batch_zeros, batch_zeros), 1)
    col1 = torch.stack((batch_zeros, cos_rot[:, 0], -sin_rot[:, 0]), 1)
    col2 = torch.stack((batch_zeros, sin_rot[:, 0], cos_rot[:, 0]), 1)
    Rmat0 = torch.stack((col0, col1, col2), 2)
    
    col0 = torch.stack((cos_rot[:, 1], batch_zeros, sin_rot[:, 1]), 1)
    col1 = torch.stack((batch_zeros, batch_ones, batch_zeros), 1)
    col2 = torch.stack((-sin_rot[:, 1], batch_zeros, cos_rot[:, 1]), 1)
    Rmat1 = torch.stack((col0, col1, col2), 2)
    
    col0 = torch.stack((cos_rot[:, 0], -sin_rot[:, 0], batch_zeros), 1)
    col1 = torch.stack((sin_rot[:, 0], cos_rot[:, 0], batch_zeros), 1)
    col2 = torch.stack((batch_zeros, batch_zeros, batch_ones), 1)
    Rmat2 = torch.stack((col0, col1, col2), 2)
    
    Rmat = torch.bmm(Rmat0, Rmat1)
    Rmat = torch.bmm(Rmat, Rmat2)
    
    transmat = trans1_2.unsqueeze(2)
    Tmat = torch.cat((Rmat, transmat), 2)
    temp_mat = torch.Tensor([0, 0, 0, 1]).to(device).unsqueeze(0)
    temp_mat = temp_mat.repeat(bs, 1).unsqueeze(1)
    
    Tmat = torch.cat((Tmat, temp_mat), 1)
    
    return Tmat

def Warp(image, depth, Tmat, K):
    bs = image.shape[0]
    height = image.shape[2]
    width = image.shape[3]
    device = image.device
    K = K.unsqueeze(0).repeat(bs, 1, 1)
    
    pix_x = torch.arange(width, device = device)
    pix_y = torch.arange(height, device = device)
    grid_x, grid_y = torch.meshgrid(pix_x, pix_y, "ij")
    pix_coord = torch.cat((grid_x, grid_y), 2)
    pix_coord = pix_coord.unsqueeze(0).repeat(bs, 1, 1, 1)
    
    mask = torch.zeros_like(image)
    
    
    
    return image, mask


if __name__ == "__main__":
    image = torch.randn((4, 3, 960, 540))       # Scale = (0.5, 0.5)
    image = F.pad(image, (2, 2, 0, 0), "constant", 0)
    depth = torch.randn((4, 1, 960, 540))       # Scale = (0.5, 0.5)
    depth = F.pad(depth, (2, 2, 0, 0), "constant", 0)
    target = torch.randn((4, 3, 960, 540))       # Scale = (0.5, 0.5)
    target = F.pad(target, (2, 2, 0, 0), "constant", 0)
    
    trans = torch.rand((4, 3))
    rot = torch.rand((4, 3))
    
    Tmat = trans_rot2Tmat(trans, rot)
    fx = 300
    fy = 300
    cx = 481
    cy = 270
    K = torch.Tensor([[fx, 0, cx],
                      [fy, 0, cy],
                      [0, 0, 1]])
    
    res_image, mask = Warp(image, depth, Tmat, K)
    target = target * mask
    
    smothl1loss = nn.SmoothL1Loss()
    loss = smothl1loss(res_image, target)
    
    print(loss)