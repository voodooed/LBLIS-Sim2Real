import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch


class KittiDataset(Dataset):
    

    def __init__(self, lidar_dir, intensity_sim_dir,incidence_dir, label_dir, intensity_real_dir, reflectance_dir,
        lidar_transform=None,incidence_transform=None,intensity_sim_transform=None,
        label_transform=None,intensity_real_transform=None, reflectance_transform=None):

        self.lidar_dir = lidar_dir
        self.intensity_sim_dir = intensity_sim_dir
        self.incidence_dir = incidence_dir
        self.label_dir = label_dir 
        self.intensity_real_dir = intensity_real_dir
        self.reflectance_dir = reflectance_dir

        self.lidar_images = os.listdir(lidar_dir)
        self.intensity_sim_images = os.listdir(intensity_sim_dir)
        self.incidence_images = os.listdir(incidence_dir)
        self.label_images = os.listdir(label_dir)
        self.intensity_real_images = os.listdir(intensity_real_dir)
        self.reflectance_images = os.listdir(reflectance_dir)

        self.lidar_transform = lidar_transform 
        self.incidence_transform = incidence_transform
        self.intensity_sim_transform = intensity_sim_transform
        self.label_transform = label_transform
        self.intensity_real_transform = intensity_real_transform
        self.reflectance_transform=reflectance_transform 

        self.length_dataset = max(len(self.intensity_sim_images ), len(self.intensity_real_images )) #Max of simulated and real dataset
        self.sim_len = len(self.intensity_sim_images)
        self.real_len = len(self.intensity_real_images)

    def __len__(self):
        return self.length_dataset #Returning the maximum length

    def __getitem__(self, index):
        lidar_path = os.path.join(self.lidar_dir, self.lidar_images[index])
        intensity_sim_path = os.path.join(self.intensity_sim_dir, self.intensity_sim_images[index])
        incidence_path = os.path.join(self.incidence_dir , self.incidence_images[index])
        label_path = os.path.join(self.label_dir , self.label_images[index])
        intensity_real_path = os.path.join(self.intensity_real_dir, self.intensity_real_images[index])
        reflectance_path = os.path.join(self.reflectance_dir , self.reflectance_images[index])
        
 

        lidar = Image.open(lidar_path).convert("L")
        intensity_sim = Image.open(intensity_sim_path).convert("L")
        incidence = Image.open(incidence_path).convert("L")
        label = Image.open(label_path).convert("L")
        intensity_real = Image.open(intensity_real_path).convert("L")
        reflectance = Image.open(reflectance_path).convert("L")

        if self.lidar_transform is not None:
            intensity_sim = self.intensity_sim_transform(intensity_sim) #For adding a channel
            lidar = self.lidar_transform(lidar)
            incidence = self.incidence_transform(incidence)
            label = self.label_transform(label)
            intensity_real = self.intensity_real_transform(intensity_real)
            reflectance = self.reflectance_transform(reflectance)

        #Change Accordingly
        simulated_img = torch.cat((lidar, incidence, reflectance), dim=0)  #in_channels=3 #T7
        
        return simulated_img, intensity_real, intensity_sim