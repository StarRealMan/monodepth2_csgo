import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

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

def Warp(target_image, source_depth, Tmat, K):
    bs = target_image.shape[0]
    height = target_image.shape[2]
    width = target_image.shape[3]
    device = target_image.device
    eps = 1e-7
    
    K_inv = torch.linalg.inv(K)
    K_inv = K_inv.unsqueeze(0).repeat(bs, 1, 1)
    K = K.unsqueeze(0).repeat(bs, 1, 1)
    
    depth = source_depth.permute(0, 1, 3, 2).contiguous().view(bs, 1, -1)
    
    pix_x = torch.arange(width, device = device)
    pix_y = torch.arange(height, device = device)
    grid_x, grid_y = torch.meshgrid(pix_x, pix_y, indexing = "xy")
    pix_coord = torch.stack((grid_x, grid_y), 2)
    pix_coord = pix_coord.unsqueeze(0).repeat(bs, 1, 1, 1)
    pix_coord = pix_coord.view(bs, -1, 2).permute(0, 2, 1)
    pix_coord = torch.cat([pix_coord, torch.ones((bs, 1, pix_coord.shape[2]))], 1)
    
    cam_coord = torch.bmm(K_inv, pix_coord)
    cam_coord = torch.mul(cam_coord, depth)
    cam_coord = torch.cat([cam_coord, torch.ones((bs, 1, cam_coord.shape[2]))], 1)
    camT_coord = torch.bmm(Tmat, cam_coord)
    
    pixT_coord = torch.bmm(K, camT_coord[:, :3, :])
    pixT_z = pixT_coord[:, 2, :].unsqueeze(1)
    pixT_coord = pixT_coord[:, :2, :] / (pixT_z + eps)
    
    mask = (pixT_coord[:, 0, :] < width) & (pixT_coord[:, 1, :] < height)
    mask = mask.view(bs, 1, height, width)
    
    pixT_coord = pixT_coord.view(bs, 2, height, width)
    pixT_coord = pixT_coord.permute(0, 2, 3, 1)
    
    pixT_coord[:, :, :, 0] /= width - 1
    pixT_coord[:, :, :, 1] /= height - 1
    pixT_coord = (pixT_coord - 0.5) * 2
    
    imageT = F.grid_sample(target_image, pixT_coord, padding_mode='zeros', align_corners = True)
    
    return imageT, mask


if __name__ == "__main__":
    
    trans = torch.rand((1, 3))
    rot = torch.rand((1, 3))
    Tmat = trans_rot2Tmat(trans, rot)
    
    target = cv2.imread("../image/test_image.jpg")
    target = torch.from_numpy(target)
    target = target.permute(2, 0, 1)
    target = target.unsqueeze(0)
    target = target.float()
    
    # should use depth of source image but not target
    depth = cv2.imread("../image/test_image_disp_raw.jpeg", cv2.IMREAD_GRAYSCALE)
    depth = torch.from_numpy(depth)
    depth = depth.unsqueeze(2)
    depth = depth.permute(2, 0, 1)
    depth = depth.unsqueeze(0)
    depth = depth.float()
    depth = 1 / (depth + 1e-7 * torch.ones_like(depth))
    
    source = cv2.imread("../image/test_image.jpg")
    source = torch.from_numpy(source)
    source = source.permute(2, 0, 1)
    source = source.unsqueeze(0)
    source = source.float()
    
    Tmat = torch.tensor([[[1.0, 0, 0, 0],
                          [0, 1.0, 0, 0],
                          [0, 0, 1.0, 0],
                          [0, 0, 0, 1.0]]])
    
    fx = 300
    fy = 300
    cx = 319
    cy = 117.5
    K = torch.Tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
    
    res_image, mask = Warp(target, depth, Tmat, K)
    
    cv2.imwrite("../image/test_image_res.jpg", res_image.squeeze(0).permute(1, 2, 0).cpu().numpy())
    
    source = torch.mul(source, mask)
    
    smothl1loss = nn.SmoothL1Loss()
    loss = smothl1loss(res_image, source)
    
    print(loss)