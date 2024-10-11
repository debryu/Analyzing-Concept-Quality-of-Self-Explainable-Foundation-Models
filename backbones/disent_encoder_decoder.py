import torch


from torch import Tensor
from torch import nn


class EncoderConv64(nn.Module):
    """
    Convolutional encoder used in beta-VAE paper for the chairs data.
    Based on row 4-6 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)

    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1):
        # checks
        (C, H, W) = x_shape
        
        super().__init__()
        if (H,W) == (64,64):
            self.z_size = z_size
            self.z_total = z_size*z_multiplier
            self.final_encoder_layer = 8
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=C, out_channels=32, kernel_size=4, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2),
                nn.ReLU(inplace=True),  # This was reverted to kernel size 4x4 from 2x2, to match beta-vae paper
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=2),
                nn.ReLU(inplace=True),  # This was reverted to kernel size 4x4 from 2x2, to match beta-vae paper
                nn.Flatten(),
                nn.Linear(in_features=1600, out_features=1600),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=1600, out_features=1600),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=1600, out_features=256),
                nn.ReLU(inplace=True),
                nn.Linear(
                    in_features=256, out_features=self.z_total
                ),  # we combine the two networks in the reference implementation and use torch.chunk(2, dim=-1) to get mu & logvar
            )
        elif (H,W) == (128,128):
            self.z_size = z_size
            self.z_total = z_size*z_multiplier
            self.final_encoder_layer = 10
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=C, out_channels=16, kernel_size=6, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=6, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=6, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=6, stride=2, padding=2),
                nn.ReLU(inplace=True),  # This was reverted to kernel size 4x4 from 2x2, to match beta-vae paper
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=6, stride=2, padding=2),
                nn.ReLU(inplace=True),  # This was reverted to kernel size 4x4 from 2x2, to match beta-vae paper
                nn.Flatten(),
                nn.Linear(in_features=2048, out_features=1024),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=1024, out_features=256),
                nn.ReLU(inplace=True),
                nn.Linear(
                    in_features=256, out_features=self.z_total
                ),  # we combine the two networks in the reference implementation and use torch.chunk(2, dim=-1) to get mu & logvar
            )       # Final dimension


        elif (H,W) == (224,224):
            self.z_size = z_size
            self.z_total = z_size*z_multiplier
            self.final_encoder_layer = 10
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=C, out_channels=16, kernel_size=4, stride=2, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=0),
                nn.ReLU(inplace=True), 
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=0),
                nn.ReLU(inplace=True),  
                nn.Flatten(),
                nn.Linear(in_features=6400, out_features=1600),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=1600, out_features=1600),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=1600, out_features=800),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=800, out_features=400),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=400, out_features=256),
                nn.ReLU(inplace=True),
                nn.Linear(
                    in_features=256, out_features=self.z_total
                ),  # we combine the two networks in the reference implementation and use torch.chunk(2, dim=-1) to get mu & logvar
            )
        else:
            raise ValueError(f"Unsupported image size for encoder: {H}x{W}")
    def forward(self, x) -> Tensor:
        # This is only used to get the MI for the Havasi paper, comment otherwise
        intermediate_out = {}
        temp = x
        for i,layer in enumerate(self.model):
            temp = layer(temp)
            #print(i,temp.shape)
            #if i <= 8:
            #    print('x:', temp.shape)
            if i == self.final_encoder_layer:
                #print('x:', temp.shape)
                intermediate_out['hidden_input'] = temp
        
        return torch.split(self.model(x), self.z_size, -1), intermediate_out
    

class EncoderConv64forBlackbox(nn.Module):
    """
    Convolutional encoder used in beta-VAE paper for the chairs data.
    Based on row 4-6 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)

    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1):
        # checks
        (C, H, W) = x_shape
        assert (H, W) == (64, 64), "This model only works with image size 64x64."
        super().__init__()

        self.z_size = z_size
        self.z_total = z_size*z_multiplier

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=128, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),  # This was reverted to kernel size 4x4 from 2x2, to match beta-vae paper
            nn.Conv2d(in_channels=384, out_channels=512, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),  # This was reverted to kernel size 4x4 from 2x2, to match beta-vae paper
            nn.Flatten(),
            nn.Linear(in_features=12800, out_features=6400),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=6400, out_features=4096
            ),  # we combine the two networks in the reference implementation and use torch.chunk(2, dim=-1) to get mu & logvar
        )

    def forward(self, x) -> Tensor:
        return torch.split(self.model(x), self.z_size, -1)
        


class DecoderConv64(nn.Module):
    """
    Convolutional decoder used in beta-VAE paper for the chairs data.
    Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)
    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1,):
        (C, H, W) = x_shape
        #assert (H, W) == (64, 64), "This model only works with image size 64x64."
        super().__init__()
        if(H,W) == (64,64):
            self.z_size = z_size
            self.z_total = z_size*z_multiplier  

            self.model = nn.Sequential(
                #nn.Linear(in_features=self.z_total, out_features=256),
                #nn.ReLU(inplace=True),
                #nn.Linear(in_features=256, out_features=1024),
                #nn.ReLU(inplace=True),
                nn.Linear(in_features=self.z_total, out_features=1024),
                nn.ReLU(inplace=True),
                nn.Unflatten(dim=1, unflattened_size=[64, 4, 4]),
                nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=32, out_channels=C, kernel_size=4, stride=2, padding=1),
            )
        elif (H,W) == (128,128):
            self.z_size = z_size
            self.z_total = z_size*z_multiplier  

            self.model = nn.Sequential(
                nn.Linear(in_features=self.z_total, out_features=256),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=256, out_features=1024),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=1024, out_features=2048),
                nn.ReLU(inplace=True),
                nn.Unflatten(dim=1, unflattened_size=[128, 4, 4]),
                nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=6, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=6, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=6, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=16, out_channels=C, kernel_size=6, stride=2, padding=2),
            )

        elif (H,W) == (224,224):
            self.z_size = z_size
            self.z_total = z_size*z_multiplier
            self.final_encoder_layer = 10
            self.model = nn.Sequential(
                nn.Linear(in_features=self.z_total, out_features=256),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=256, out_features=6400),
                nn.ReLU(inplace=True),
                #nn.Linear(in_features=400, out_features=800),
                #nn.ReLU(inplace=True),
                #nn.Linear(in_features=800, out_features=1600),
                #nn.ReLU(inplace=True),
                #nn.Linear(in_features=1600, out_features=3200),
                #nn.ReLU(inplace=True),
                #nn.Linear(in_features=3200, out_features=6400),
                #nn.ReLU(inplace=True),
                nn.Unflatten(dim=1, unflattened_size=[256, 5, 5]),
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=0),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=0),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=0),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=16, out_channels=C, kernel_size=4, stride=2, padding=0),
            )

        else:
            raise ValueError(f"Unsupported image size for decoder: {H}x{W}")

    def forward(self, z) -> Tensor:
        return self.model(z)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #