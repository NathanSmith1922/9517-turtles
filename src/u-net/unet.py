import torch
import torch.nn as nn

"""
References

[1] N. Tomar, “UNET Implementation in PyTorch - Idiot Developer,” Idiot Developer, May 22, 2021. https://idiotdeveloper.com/unet-implementation-in-pytorch/ (accessed Nov. 06, 2024).
[2] M. Tran, “Understanding U-Net,” Medium, Nov. 16, 2022. https://towardsdatascience.com/understanding-u-net-61276b10f360
[3] R. Chew, “U-NETS For Dummies (PyTorch & TensorFlow) - Ryan Chew - Medium,” Medium, Sep. 04, 2024. https://medium.com/@chewryan0/u-nets-for-dummies-pytorch-tensorflow-dddcdb8a2759 (accessed Nov. 07, 2024).
[4] “torch — PyTorch 1.12 documentation,” pytorch.org. https://pytorch.org/docs/stable/torch.html

"""

LEARNING_RATE = 0.001

class Conv3x3ReLU(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        # Using padding after the convolutional layer will keep the image 
        # size constant until the pooling at the end of the block.

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, inputs):
        return self.conv(inputs)
    
class Encoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = Conv3x3ReLU(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p
    
class Decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2)
        self.conv = Conv3x3ReLU(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, layers=4):
        super().__init__()

        self.encoders = nn.ModuleList()
        # Create all the Encoders
        current_in_channels = in_channels
        for i in range(layers):
            # Doubling the number of output channels at each layer
            current_out_channels = 64 * (2**i)
            self.encoders.append(Encoder(current_in_channels, current_out_channels))
            current_in_channels = current_out_channels

        self.bottleneck = Conv3x3ReLU(current_in_channels, current_in_channels * 2)

        self.decoders = nn.ModuleList()
        # Create all the Decoders
        for i in range(layers, 0, -1):
            current_in_channels = 64 * (2**i)
            current_out_channels = 64 * (2**(i-1))
            self.decoders.append(Decoder(current_in_channels, current_out_channels))
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

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
    
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    unet = UNet(1, 2, 4).to(device)
    print(unet)
    optimizer = torch.optim.Adam(params=unet.parameters(), lr=LEARNING_RATE)
    

if __name__ == "__main__":
    main()
 