from transform_utils import lidar_transform,intensity_real_transform,incidence_transform,reflectance_transform
import os
import torch
from PIL import Image
from generator import Generator
from torchvision.utils import save_image

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_image(lidar_path, incidence_path, reflectance_path, output_folder, model, device):
    # Load the input images
    lidar = Image.open(lidar_path).convert("L")
    incidence = Image.open(incidence_path).convert("L")
    reflectance = Image.open(reflectance_path).convert("L")

    # Apply your transformations here...

    lidar = lidar_transform(lidar)
    incidence = incidence_transform(incidence)
    reflectance = reflectance_transform(reflectance) 

    def inverse_transform(tensor):
        mean = 0.4257
        std = 0.2276
        inv_tensor = tensor * std + mean
        return inv_tensor

    #Voxelscape
    input_data = torch.cat((lidar, incidence, reflectance), dim=0) #T7 #in_channels=3
    

    input_data = input_data.unsqueeze(0) 

    input_data = input_data.to(device)
    
    with torch.no_grad():
        output = model(input_data)

    output = inverse_transform(output)
    
    # Save the output tensor as an image
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(lidar_path))[0] +'.jpg')
    save_image(output, output_file)

folder = "val"

# specify your input and output directories
input_folder = "/home/viveka21/projects/def-rkmishra-ab/viveka21/VoxelScape_Data/Val/" #Change accordingly
output_folder = f"/home/viveka21/projects/def-rkmishra-ab/viveka21/VoxelScape_Data/Val/{folder}_lidar_cycle_intensity" #Change accordingly

if not os.path.exists(output_folder):
        os.makedirs(output_folder)

in_channels = 3 #Change accordingly

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Generator(img_channels=in_channels, out_channels=1)  # Initialize your model architecture
checkpoint = torch.load('/home/viveka21/projects/def-rkmishra-ab/viveka21/Objective 3/CycleGAN/Output_1.0/T7/Model/gen_r.pth.tar_T7')

model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()

# iterate over the subdirectories inside the input folder
subfolders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
# Assuming all images have same filenames across subfolders, pick one (say "test_lidar_depth") as reference
reference_subfolder = f"{folder}_lidar_depth"
reference_files = os.listdir(os.path.join(input_folder, reference_subfolder))

for filename in reference_files:
    if filename.endswith(".jpg"):
        lidar_path = os.path.join(input_folder, f"{folder}_lidar_depth", filename)
        incidence_path = os.path.join(input_folder, f"{folder}_incidence_mask", filename)
        reflectance_path = os.path.join(input_folder, f"{folder}_lidar_reflectance", filename)
        
        process_image(lidar_path, incidence_path, reflectance_path, output_folder, model, device)