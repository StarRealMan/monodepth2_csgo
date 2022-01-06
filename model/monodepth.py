import torch
import torch.nn as nn
import torch.nn.functional as F

class DownSampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel = None):
        super(DownSampleBlock, self).__init__()
        if out_channel == None:
            out_channel = int(in_channel * 2)

        self.main = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, input):
        output = self.main(input)

        return output

class UpSampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel = None):
        super(UpSampleBlock, self).__init__()
        if out_channel == None:
            out_channel = int(in_channel * 0.5)
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
    
    def forward(self, input):
        output = self.main(input)

        return output


class DepthNet(nn.Module):
    def __init__(self, nc, ngf):
        super(DepthNet, self).__init__()
        self.depth = 4

        self.startconv = nn.Sequential(
            nn.Conv2d(nc, ngf, 7, 1, 3, bias = False),
            nn.LeakyReLU(0.2),
        )
        
        self.DownSampleList = nn.ModuleList()
        channel = ngf
        down_channel_list = []
        for depth_num in range(self.depth):
            downSample = DownSampleBlock(channel)
            down_channel_list.append(channel)
            channel = channel * 2
            self.DownSampleList.append(downSample)

        self.UpSampleList = nn.ModuleList()
        for depth_num in range(self.depth):
            next_channel = int(channel * 0.5)
            upsample = UpSampleBlock(channel, next_channel)
            channel = next_channel + down_channel_list[self.depth  - depth_num - 1]
            self.UpSampleList.append(upsample)
        
        self.endconv = nn.Sequential(
            nn.Conv2d(5 * ngf, 2 * ngf, 3, 1, 1, bias = False),
            nn.BatchNorm2d(2 * ngf),
            nn.ReLU(),
            nn.Conv2d(2 * ngf, ngf, 3, 1, 1, bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            nn.Conv2d(ngf, 1, 3, 1, 1, bias = False),
            nn.Tanh(),
        )

        self.down_result_list = []

    def forward(self, input):
        x = self.startconv(input)
        self.down_result_list.clear()

        for depth_num  in range(len(self.DownSampleList)):
            net = self.DownSampleList[depth_num]
            self.down_result_list.append(x)
            x = net(x)
        
        for depth_num  in range(len(self.UpSampleList)):
            net = self.UpSampleList[depth_num]
            x = net(x)
            skip = self.down_result_list[self.depth - depth_num - 1]
            x = torch.cat((x, skip), 1)

        output = self.endconv(x)

        return output

class PoseNet(nn.Module):
    def __init__(self, nc, ndf):
        super(PoseNet, self).__init__()
        depth = 2
        self.ndf = ndf

        self.t_anchor_len = 12
        self.t_bin_size = 2 * torch.pi / self.t_anchor_len
        self.t_anchor_step = 1
        self.t_anchor_var = 1

        self.r_anchor_len = 6
        self.r_bin_size = torch.pi / self.r_anchor_len

        self.startconv = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
        )

        channel = ndf
        self.DownSampleList = nn.ModuleList()
        for depth_num in range(depth):
            downsample = DownSampleBlock(channel)
            channel = channel * 2
                
            self.DownSampleList.append(downsample)

        self.endconv = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),
        )

        self.fused_conv = nn.Sequential(
            nn.Conv2d(ndf * 16, ndf * 16, 4, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(ndf * 16, ndf * 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 16, ndf * 16, 3, 2, 1, bias=False),
            nn.Sigmoid(),
        )

        self.translation_head = nn.Sequential(
            nn.Linear(ndf * 32, ndf * 8),
            nn.ReLU(),
            nn.Linear(ndf * 8, 3 + self.t_anchor_len * 3),
            nn.ReLU(),
            nn.Tanh()
        )
        
        self.rotation_head = nn.Sequential(
            nn.Linear(ndf * 32, ndf * 8),
            nn.ReLU(),
            nn.Linear(ndf * 8, 3 + self.r_anchor_len * 3),
            nn.ReLU(),
            nn.Tanh()
        )

        self.trans_amp_head = nn.Sequential(
            nn.Linear(ndf * 32, ndf * 8),
            nn.ReLU(),
            nn.Linear(ndf * 8, 1),
            nn.ReLU(),
            nn.Tanh()
        )

    def forward(self, input1, input2):

        x1 = self.startconv(input1)
        for depth_num  in range(len(self.DownSampleList)):
            net = self.DownSampleList[depth_num]
            x1 = net(x1)
        x1 = self.endconv(x1)

        x2 = self.startconv(input2)
        for depth_num  in range(len(self.DownSampleList)):
            net = self.DownSampleList[depth_num]
            x2 = net(x2)
        x2 = self.endconv(x2)

        fused = torch.cat((x1, x2), 1)
        fused = self.fused_conv(fused)
        fused = fused.view(-1, 32 * self.ndf)

        fused_trans = self.translation_head(fused)
        fused_rot = self.rotation_head(fused)

        t_bin = fused_trans[:, 3:].view(-1, 3, self.t_anchor_len)
        r_bin = fused_rot[:, 3:].view(-1, 3, self.r_anchor_len)

        trans = self.t_bin_size * (fused_trans[:, :3] + torch.argmax(t_bin, 2))
        rot = self.r_bin_size * (fused_rot[:, :3] + torch.argmax(r_bin, 2))
        
        trans_norm = torch.unsqueeze(torch.norm(trans, dim = 1), 1)
        trans = torch.div(trans, trans_norm)
        
        
        trans_amp = self.t_anchor_step + self.t_anchor_var * self.trans_amp_head(fused)

        return trans * trans_amp, rot

if __name__ == "__main__":
    net = PoseNet(3, 32)

    input1 = torch.randn((4, 3, 960, 540))       # Scale = (0.5, 0.5)
    input1 = F.pad(input1, (2, 2, 0, 0), "constant", 0)
    input2 = torch.randn((4, 3, 960, 540))       # Scale = (0.5, 0.5)
    input2 = F.pad(input2, (2, 2, 0, 0), "constant", 0)

    trans, rot = net(input1, input2)
    
    print(trans)
    print(rot)