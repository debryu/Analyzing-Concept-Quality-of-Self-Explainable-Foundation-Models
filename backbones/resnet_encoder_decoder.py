import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
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
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        print(self.model)
        self.model.eval()
        self.z_size = z_size
        self.z_total = z_size*z_multiplier
        self.bottleneck = nn.Sequential(
            nn.Linear(in_features=1000, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=self.z_total),
        )
        '''
        self.preprocess = transforms.Compose([
                            transforms.Resize(224),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            transforms.ToTensor(),
                            ])
        '''

        if (H,W) == (224,224):
            self.z_size = z_size
            self.z_total = z_size*z_multiplier
            self.final_encoder_layer = 10
        else:
            raise ValueError(f"Unsupported image size for encoder: {H}x{W}")
    def forward(self, x) -> Tensor:
        # This is only used to get the MI for the Havasi paper, comment otherwise
        #print(x.shape)
        #x = self.preprocess(x)
        #print(x.shape)

        with torch.no_grad():
            intermediate_out = self.model(x)
        #print(intermediate_out.shape)
        bottleneck_out = self.bottleneck(intermediate_out)
        return torch.split(bottleneck_out, self.z_size, -1), {'hidden_input':intermediate_out}
    


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
                nn.Linear(in_features=self.z_total, out_features=256),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=256, out_features=1024),
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
                nn.Linear(in_features=256, out_features=400),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=400, out_features=800),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=800, out_features=1600),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=1600, out_features=3200),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=3200, out_features=6400),
                nn.ReLU(inplace=True),
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