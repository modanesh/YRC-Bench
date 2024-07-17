import os
import torch
from PIL import Image
import h5py
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


def global_contrast_normalization(image, scale = "l1"):
    mean = np.mean(image)
    image -= mean
    if scale == "l1":
        contrast = torch.mean(torch.abs(image))
    else:
        num_features = int(np.prod(image.shape))
        contrast = torch.sqrt(torch.sum(image ** 2)) / num_features
    image /= contrast
    return image


def preprocess_and_save_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok = True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: global_contrast_normalization(x, scale = "l1"))
    ])
    with h5py.File(input_dir, "r") as f:
        for key in f.keys():
            obs = f[key][:]  # each observation is stored as its own unit, same hierarchy level
            img_tensor = transform(obs)
            img_normalized = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
            torch.save(img_normalized, os.path.join(output_dir, f"{key}.pt"))


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform = None, target_transform = None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_files = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith(".pt")]  # load already preprocessed files
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = torch.load(img_path)
        if self.transform:
            image = self.transform(image)
        label = 0
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, idx


def get_dataloader(dataset, batch_size, shuffle = True, num_workers = 0):
    # transform = transforms.Compose([
    #     transforms.ToTensor()
    # ])
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return dataloader
