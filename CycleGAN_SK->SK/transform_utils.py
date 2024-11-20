from torchvision import transforms

binary_transform = transforms.Compose([
    transforms.ToTensor(),
    
])

lidar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1319], std=[0.2112])
])


intensity_sim_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3619], std=[0.3233])
])

incidence_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4144], std=[0.3529])
])



label_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1607], std=[0.3530])
])


intensity_real_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4257], std=[0.2276])
])

reflectance_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2697], std=[0.4466])
])