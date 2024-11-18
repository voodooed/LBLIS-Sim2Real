from torchvision import transforms

binary_transform = transforms.Compose([
    transforms.ToTensor(),
])

lidar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1218], std=[0.1839])
])


intensity_sim_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2482], std=[0.2179])
])


incidence_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4776], std=[0.2956])
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
    transforms.Normalize(mean=[0.2728], std=[0.5145])
])
