import os
import torch
import torchvision
from dataset import KittiDataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import config
from torchvision.utils import save_image
import math

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_outputs(gen, val_loader, epoch, folder, num_images=10):
    gen.eval()

    sim, real, phy = next(iter(val_loader))
    sim = sim.to(config.DEVICE)
    real = real.to(config.DEVICE) 
    phy = phy.to(config.DEVICE) 

    with torch.no_grad():
        fake_real = gen(sim)
        fake_real_denormalized = (fake_real * 0.2276) + 0.4257  # Denormalize generated fake images
        sim_denormalized = (phy * 0.2179) + 0.2482  # Denormalize ground truth physics intensity images

    # Create a new directory for this epoch
    epoch_dir = os.path.join(folder, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)

    # Save each image individually
    for i in range(min(num_images, sim.size(0))):

        # Get the image names from the val_loader
        image_name = os.path.splitext(val_loader.dataset.lidar_images[i])[0]

        # Create the filenames for fake and true images
        fake_real_image_name = f"{image_name}_generated_intensity.png"
        sim_image_name = f"{image_name}_sim_intensity.png"

        # Save the denormalized fake and true images using torchvision's save_image
        save_image(fake_real_denormalized[i], os.path.join(epoch_dir, fake_real_image_name))
        save_image(sim_denormalized[i], os.path.join(epoch_dir, sim_image_name))

    gen.train()



def get_loaders(
    train__lidar_dir,
    train_intensity_sim_dir,
    train_incidence_dir,
    train_label_dir,
    train_intensity_real_dir,
    train_reflectance_dir,

    val_lidar_dir,
    val_intensity_sim_dir,
    val_incidence_dir,
    val_label_dir,
    val_intensity_real_dir,
    val_reflectance_dir,



    batch_size,
    lidar_transform,
    incidence_transform,
    intensity_sim_transform,
    label_transform,
    intensity_real_transform,
    reflectance_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = KittiDataset(
        lidar_dir = train__lidar_dir,
        intensity_sim_dir=train_intensity_sim_dir,
        incidence_dir=train_incidence_dir,
        label_dir = train_label_dir,
        intensity_real_dir=train_intensity_real_dir,
        reflectance_dir = train_reflectance_dir,

        lidar_transform=lidar_transform,
        incidence_transform=incidence_transform,
        intensity_sim_transform=intensity_sim_transform,
        label_transform=label_transform,
        intensity_real_transform=intensity_real_transform,
        reflectance_transform=reflectance_transform,
       
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = KittiDataset(
        lidar_dir = val_lidar_dir,
        intensity_sim_dir = val_intensity_sim_dir,
        incidence_dir = val_incidence_dir,
        label_dir = val_label_dir,
        intensity_real_dir = val_intensity_real_dir,
        reflectance_dir = val_reflectance_dir,


        lidar_transform=lidar_transform,
        incidence_transform=incidence_transform,
        intensity_sim_transform=intensity_sim_transform,
        label_transform=label_transform,
        intensity_real_transform=intensity_real_transform,
        reflectance_transform=reflectance_transform,

    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader