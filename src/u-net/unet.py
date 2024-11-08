import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader, random_split, default_collate

"""
Coding References

[1] N. Tomar, “UNET Implementation in PyTorch - Idiot Developer,” Idiot Developer, May 22, 2021. https://idiotdeveloper.com/unet-implementation-in-pytorch/ (accessed Nov. 06, 2024).
[2] M. Tran, “Understanding U-Net,” Medium, Nov. 16, 2022. https://towardsdatascience.com/understanding-u-net-61276b10f360
[3] R. Chew, “U-NETS For Dummies (PyTorch & TensorFlow) - Ryan Chew - Medium,” Medium, Sep. 04, 2024. https://medium.com/@chewryan0/u-nets-for-dummies-pytorch-tensorflow-dddcdb8a2759 (accessed Nov. 07, 2024).
[4] “torch — PyTorch 1.12 documentation,” pytorch.org. https://pytorch.org/docs/stable/torch.html
[5] FernandoPC25, “Mastering U-Net: A Step-by-Step Guide to Segmentation from Scratch with PyTorch,” Medium, Apr. 25, 2024. https://medium.com/@fernandopalominocobo/mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114

"""

class TurtlesData(Dataset):
    def __init__(self, annotations_file='../data/annotations.json', root_directory='../data/', transform=None):
        self.root_dir = root_directory
        self.coco = COCO(annotations_file)
        self.transform = transform

        self.image_ids = self.coco.getImgIds()
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image = np.array(Image.open(self.root_dir + image_info['file_name']))

        anns_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(anns_ids)

        if len(anns) == 0:
            mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        else:
            mask = self.coco.annToMask(anns[0])
            for i in range(1, len(anns)):
                mask += self.coco.annToMask(anns[i])
        
        if (self.transform):
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
    
# ---------------------------------------------------------------------------------------
    
class Conv3x3ReLU(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )

    def forward(self, inputs):
        return self.conv(inputs)
    
class Encoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = Conv3x3ReLU(in_c, out_c)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p
    
class Decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = Conv3x3ReLU(in_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)

        # Skip dimensions are matched with x, if they're both different.
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            skip = skip[:, :, :x.size(2), :x.size(3)]
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        return x

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, layers=4):
        super().__init__()

        current_in_channels = in_channels
        current_out_channels = 64

        self.encoders = nn.ModuleList()
        for _ in range(layers):
            self.encoders.append(Encoder(current_in_channels, current_out_channels))
            current_in_channels = current_out_channels
            current_out_channels = current_in_channels * 2

        self.bottleneck = Conv3x3ReLU(current_in_channels, current_out_channels)

        self.decoders = nn.ModuleList()
        for _ in range(layers):
            current_in_channels = current_out_channels
            current_out_channels = current_in_channels // 2
            self.decoders.append(Decoder(current_in_channels, current_out_channels))
        
        self.final_conv = nn.Conv2d(current_out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)
        
        x = self.bottleneck(x)
        
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips[-(i + 1)])        
        x = self.final_conv(x)
        return x

LEARNING_RATE = 3e-4
BATCH_SIZE = 8
EPOCHS = 1
    
def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])
    train_dataset = TurtlesData(transform=transform)
    generator = torch.Generator().manual_seed(25)
    train_dataset, test_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)
    test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5], generator=generator)

    # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # This allows us to use CUDA supported GPU's for faster training

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        num_workers = torch.cuda.device_count() * 4
    torch.cuda.empty_cache()

    train_dataloader = DataLoader(dataset=train_dataset, num_workers=num_workers, pin_memory=False, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=train_dataset, num_workers=num_workers, pin_memory=False, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset=train_dataset, num_workers=num_workers, pin_memory=False, batch_size=BATCH_SIZE, shuffle=True)

    # Example usage would be grayscale images with 1 colour channel, 
    # 3 turtle segmentation classes for our output channel with 4 training layers.
    model = UNet(3, 3, 4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn  = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(EPOCHS):
        for idx, (img, mask) in enumerate(train_dataloader):
            img = img.to(device)
            target = mask.to(device)
            target = target.squeeze(1).long()
            
            logits = model(img)
            optimizer.zero_grad()

            loss = loss_fn(logits, target)
            loss.backward()

if __name__ == "__main__":
    main()
 