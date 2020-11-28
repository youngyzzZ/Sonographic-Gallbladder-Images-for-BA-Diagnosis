# @ FileName: gallbladder_dataset.py
# @ Author: Alexis
# @ Time: 20-11-28 下午9:17

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import SimpleITK as sitk
import numpy as np


class GallbladderDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.frame = pd.read_csv(csv_file, encoding='utf-8', header=None)
        self.root_dir = root_dir
        # print('csv_file source----->', csv_file)
        # print('root_dir source----->', root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.frame.iloc[idx, 0])
        img_name = os.path.basename(img_path)
        # print(img_name)
        _, extension = os.path.splitext(self.frame.iloc[idx, 0])
        # print(extension)
        image = self.image_loader(img_path, extension)
        # print(image)
        label = int(self.frame.iloc[idx, 1])
        if self.transform is not None:
            image = self.transform(image)
        sample = {'image': image, 'label': label, 'img_name': img_name}
        return sample

    def image_loader(self, img_name, extension):
        if extension == '.JPG':
            # print('读取jpg')
            return self.read_jpg(img_name)
        elif extension == '.jpg':
            # print('读取jpg')
            return self.read_jpg(img_name)
        elif extension == '.DCM':
            # print('读取dcm')
            return self.read_dcm(img_name)
        elif extension == '.dcm':
            # print('读取dcm')
            return self.read_dcm(img_name)
        elif extension == '.Bmp':
            # print('读取Bmp')
            return self.read_bmp(img_name)
        elif extension == '.png':
            return self.read_png(img_name)

    def read_jpg(self, img_name):
        return Image.open(img_name)

    def read_dcm(self, img_name):
        ds = sitk.ReadImage(img_name)
        img_array = sitk.GetArrayFromImage(ds)
        img_bitmap = Image.fromarray(img_array[0])
        return img_bitmap

    def read_bmp(self, img_name):
        return Image.open(img_name)

    def read_png(self, img_name):
        return Image.open(img_name)


if __name__ == '__main__':
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = GallbladderDataset(
        csv_file='../../../dataset/final_train_cl/siggraph17/label_contain_empty/label_contain_empty_all.csv',
        root_dir='../../../dataset/final_train_cl/siggraph17',
        transform=tf
    )
    dataloader = DataLoader(dataset=dataset, batch_size=int(len(dataset) * 0.1), shuffle=False, num_workers=4)
    for item in dataloader:
        images = item['image']
        images = images.numpy()
        mean = np.mean(images, axis=(0, 2, 3))
        std = np.std(images, axis=(0, 2, 3))
        break
    print(mean, std)
