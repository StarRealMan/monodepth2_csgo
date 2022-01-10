import os
import sys
sys.path.append("..")
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..model import monodepth
from ..model import loss
from ..dataset import csgo_dataset

arg_batchsize = 4
arg_workers = 8
arg_epochs = 10
arg_lr = 1e-4
arg_b1 = 0.5
arg_b2 = 0.999
arg_nf = 32
arg_lambda = 0.1

arg_image_shape = (235, 638)

arg_fx = 300.0
arg_fy = 300.0
arg_cx = arg_image_shape[0]/2.0
arg_cy = arg_image_shape[1]/2.0


arg_K = torch.Tensor([[arg_fx, 0, arg_cx],
                      [0, arg_fy, arg_cy],
                      [0, 0, 1]])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(3)

depthnet = monodepth.DepthNet(3, arg_nf).to(device)
posenet = monodepth.PoseNet(3, arg_nf).to(device)

monodepth.weight_init(depthnet)
monodepth.weight_init(depthnet)

depthnet_optim = optim.Adam(depthnet.parameters(), lr = arg_lr, betas = (arg_b1, arg_b2))
posenet_optim = optim.Adam(posenet.parameters(), lr = arg_lr, betas = (arg_b1, arg_b2))

recon_loss = loss.Loss(arg_K)
pose_t_loss = nn.MSELoss()
pose_r_loss = nn.MSELoss()

dataroots = ["../data/Inferno_1", "../data/Inferno_2", "../data/Inferno_3",
             "../data/Mirage_1", "../data/Mirage_2", "../data/Mirage_3",
             "../data/Nuke_1", "../data/Nuke_2", "../data/Nuke_3",
             "../data/Overpass_1", "../data/Overpass_2", "../data/Overpass_3",
             "../data/Train_1", "../data/Train_2", "../data/Train_3",
             "../data/Vertigo_1", "../data/Vertigo_2", "../data/Vertigo_3"]

top_clip = [3, 2, 3, 
            2, 2, 4,
            2, 2, 2,
            3, 3, 2,
            2, 2, 2,
            4, 2, 2]

bottom_clip = [9, 7, 17,
               8, 6, 6,
               8, 7, 6,
               6, 7, 7,
               8, 6, 16,
               6, 7, 6]

lens = [775, 797, 705,
        557, 568, 557,
        514, 491, 538,
        617, 621, 538,
        445, 480, 643,
        421, 503, 612]

dataset = csgo_dataset.CSGODataset(arg_K, arg_image_shape, dataroots, top_clip, bottom_clip, lens)
dataloader = DataLoader(dataset, batch_size = arg_batchsize, shuffle=True, num_workers = arg_workers)

print('train is starting')
depthnet = depthnet.train()
posenet = posenet.train()

for epoch in range(arg_epochs):
    losses = 0
    
    for image_last, image_this, image_next in dataloader:
        t = time.time()
        
        depth_this = depthnet(image_this)
        
        last_trans, last_rot = posenet(image_this, image_last)
        next_trans, next_rot = posenet(image_this, image_next)
        
        last_loss = recon_loss(image_last, image_this, depth_this, last_trans, last_rot)
        next_loss = recon_loss(image_next, image_this, depth_this, next_trans, next_rot)
        pose_consistant_loss = pose_t_loss(last_trans, next_trans) + pose_r_loss(last_rot, next_rot)
        final_loss = min(last_loss, next_loss) + arg_lambda * pose_consistant_loss
        
        losses += final_loss
        
        depthnet_optim.zero_grad()
        posenet_optim.zero_grad()
        
        final_loss.backward()
        depthnet_optim.step()
        posenet_optim.step()
        
    print(f'[{epoch + 1}/{arg_epochs}]\tloss : {losses:.6f}\ttime : {time.time() - t:.3f}s')
    
torch.save(depthnet_optim.state_dict, '../models/depthnet/final_model.pt')
torch.save(posenet_optim.state_dict, '../models/posenet/final_model.pt')