import torch
import os

Trial_Num = "T1" #Change
Output_Trial_Num = "Output_3.0"
Trial_Path = f"/home/viveka21/projects/def-rkmishra-ab/viveka21/Objective 3/CycleGAN/{Output_Trial_Num}/{Trial_Num}/Model" #Change
if not os.path.exists(Trial_Path):
    os.makedirs(Trial_Path)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IN_CHANNELS_R = 1 #Input channel for real intensity
IN_CHANNELS_S = 3 #Input channel for simulated data - Depth, IA, MR
OUT_CHANNELS = 1

LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
LAMBDA_Physics = 10
BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True
NUM_EPOCHS = 100 #Change
LOAD_MODEL = False #Change
SAVE_MODEL = True


CHECKPOINT_GEN_S = f"{Trial_Path}/gen_s.pth.tar_{Trial_Num}"
CHECKPOINT_GEN_R = f"{Trial_Path}/gen_r.pth.tar_{Trial_Num}"
CHECKPOINT_DISC_S = f"{Trial_Path}/disc_s.pth.tar_{Trial_Num}"
CHECKPOINT_DISC_R = f"{Trial_Path}/disc_r.pth.tar_{Trial_Num}"

OUTPUT_FOLDER = f"/home/viveka21/projects/def-rkmishra-ab/viveka21/Objective 3/CycleGAN/{Output_Trial_Num}/{Trial_Num}/Output" #Change
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

#Input Directory

base_path = "/home/viveka21/projects/def-rkmishra-ab/viveka21/VoxelScape_Data/"

TRAIN_Lidar_DIR = base_path + "Train/train_lidar_depth"
TRAIN_Intensity_Sim_DIR = base_path + "Train/train_lidar_intensity"
TRAIN_Incidence_DIR = base_path + "Train/train_incidence_mask"
TRAIN_LABEL_DIR = base_path + "Train/train_lidar_label"
TRAIN_Intensity_Real_DIR = base_path + "Train/train_lidar_real_intensity_nuscenes"
TRAIN_REFLECTANCE_DIR = base_path + "Train/train_lidar_reflectance"

VAL_Lidar_DIR = base_path + "Val/val_lidar_depth"
VAL_Intensity_Sim_DIR = base_path + "Val/val_lidar_intensity"
VAL_Incidence_DIR = base_path + "Val/val_incidence_mask"
VAL_LABEL_DIR = base_path + "Val/val_lidar_label"
VAL_Intensity_Real_DIR = base_path + "Val/val_lidar_real_intensity_nuscenes"
VAL_REFLECTANCE_DIR = base_path + "Val/val_lidar_reflectance"

