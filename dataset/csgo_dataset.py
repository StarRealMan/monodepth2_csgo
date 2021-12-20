import os
import numpy as np
import cv2

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as vutils


class CSGODataset(data.Dataset):
    def __init__(self, K, shape, dataroots, top_clips, bottom_clips, lens):
        self.K = K
        self.full_res_shape = (1920, 1080)
        self.resized_shape = shape
        self.dataroots = dataroots
        self.top_clips = top_clips
        self.lens = lens
        self.full_len = 0
        self.steps = []
        self.scale = (self.resized_shape[0] / self.full_res_shape[0],
                      self.resized_shape[1] / self.full_res_shape[1])
    
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.resized_shape),
            transforms.CenterCrop(self.resized_shape),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        for i, len in enumerate(self.lens):
            self.steps.append(self.full_len)
            self.full_len += len - 2 - top_clips[i] - bottom_clips[i]

    def index2path(self, index):
        for i, step in enumerate(self.steps):
            if index < step:
                path = self.dataroots[i- 1]
                num = index - self.steps[i - 1] + self.top_clips[i] + 1
                break
        
        return path, num;

    def path2torch(self, path, images_id):
        torches = []
        for image_id in images_id:
            image_path = os.path.join(path, str(image_id) + ".jpg")
            image = cv2.imread(image_path)
            image_torch = self.transform(image)
            torches.append(image_torch)
        
        return torches

    def __len__(self):
        return self.full_len
    
    def __getitem__(self, index):
        path, num = self.index2path(index)
        images_id = [num - 1, num, num + 1]
        image_last, image_this, image_next = self.path2torch(path, images_id)

        return image_last, image_this, image_next



def video2image(data_path, frame_draw = 0.2):

    folder_name = data_path.split('/')[-1].split('.')[0] + "/"
    parent_names = data_path.split('/')[:-1]
    parent_name = ""
    for name in parent_names:
        parent_name += name + "/"

    child_name = os.path.join(parent_name, folder_name)

    if not os.path.exists(child_name):
        os.makedirs(child_name)

    image_num = 0
    count = 0
    cap = cv2.VideoCapture(data_path)

    print("Saving at: ", child_name)

    while(cap):
        ret, frame = cap.read()
        if ret == False:
            break

        count += frame_draw
        if count >= 1:
            cv2.imwrite(child_name + str(image_num) + ".jpg", frame)
            image_num += 1
            count = 0

    cap.release()

if __name__ == "__main__":

    dataroots = ["../data/Inferno_1", "../data/Inferno_2", "../data/Inferno_3",
                 "../data/Mirage_1", "../data/Mirage_2", "../data/Mirage_3",
                 "../data/Nuke_1", "../data/Nuke_2", "../data/Nuke_3",
                 "../data/Overpass_1", "../data/Overpass_2", "../data/Overpass_3",
                 "../data/Train_1", "../data/Train_2", "../data/Train_3",
                 "../data/Vertigo_1", "../data/Vertigo_2", "../data/Vertigo_3"]

    # for dataroot in dataroots:
    #     video_name = dataroot + ".mp4"
    #     video2image(video_name)

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

    dataset = CSGODataset(None, (1024, 512), dataroots, top_clip, bottom_clip, lens)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size = 4, shuffle = True, num_workers = 8)

    # for data in dataloader:
    #     image_last, image_this, image_next = data
    #     print(image_last.shape)
    #     break

    num = 0
    image_last, image_this, image_next = dataset.__getitem__(num)
    print(image_last.shape)
        
