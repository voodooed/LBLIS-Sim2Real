import torch
import os

Trial_Num = "T2" #Change
Trial_Path = f"/home/viveka21/projects/def-rkmishra-ab/viveka21/Objective 3/CycleGAN/Output_1.0/{Trial_Num}/Model" #Change
if not os.path.exists(Trial_Path):
    os.makedirs(Trial_Path)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IN_CHANNELS_R = 1 #Input channel for real intensity
IN_CHANNELS_S = 2 #Input channel for simulated data - Depth, IA, Label
OUT_CHANNELS = 1

LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
LAMBDA_Physics = 0.0
BATCH_SIZE = 8
NUM_WORKERS = 3
PIN_MEMORY = True
NUM_EPOCHS = 200 #Change
LOAD_MODEL = False
SAVE_MODEL = True


CHECKPOINT_DISC_LOAD = "/DATA2/Vivek/Code/Implementation/GAN/Pix2Pix_3.0/T3/disc.pth.tar_T3_epoch_50" #Change
CHECKPOINT_GEN_LOAD = "/DATA2/Vivek/Code/Implementation/GAN/Pix2Pix_3.0/T3/gen.pth.tar_T3_epoch_50" #Change

CHECKPOINT_GEN_S = f"{Trial_Path}/gen_s.pth.tar_{Trial_Num}"
CHECKPOINT_GEN_R = f"{Trial_Path}/gen_r.pth.tar_{Trial_Num}"
CHECKPOINT_DISC_S = f"{Trial_Path}/disc_s.pth.tar_{Trial_Num}"
CHECKPOINT_DISC_R = f"{Trial_Path}/disc_r.pth.tar_{Trial_Num}"

OUTPUT_FOLDER = f"/home/viveka21/projects/def-rkmishra-ab/viveka21/Objective 3/CycleGAN/Output_1.0/{Trial_Num}/Output" #Change
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

#Input Directory

base_path = "/DATA2/Vivek/Vivekk/Data/VoxelScape/Data/"

TRAIN_Lidar_DIR = base_path + "Train/train_lidar_depth"
TRAIN_Intensity_Sim_DIR = base_path + "Train/train_lidar_intensity"
TRAIN_Incidence_DIR = base_path + "Train/train_incidence_mask"
TRAIN_LABEL_DIR = base_path + "Train/train_lidar_label"
TRAIN_Intensity_Real_DIR = base_path + "Train/train_lidar_real_intensity"
TRAIN_REFLECTANCE_DIR = base_path + "Train/train_lidar_reflectance"

VAL_Lidar_DIR = base_path + "Val/val_lidar_depth"
VAL_Intensity_Sim_DIR = base_path + "Val/val_lidar_intensity"
VAL_Incidence_DIR = base_path + "Val/val_incidence_mask"
VAL_LABEL_DIR = base_path + "Val/val_lidar_label"
VAL_Intensity_Real_DIR = base_path + "Val/val_lidar_real_intensity"
VAL_REFLECTANCE_DIR = base_path + "Val/val_lidar_reflectance"

