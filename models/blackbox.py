import torch
from utils.args import *
from utils.conf import get_device
from models.vaesup import VAESUP
from utils.vae_loss import betaGlanceNet_Loss
from utils.losses import Blackbox_Loss

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Blackbox model')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class Blackbox(torch.nn.Module):
    NAME = 'blackbox'
    '''
    Full blackbox model
    '''
    def __init__(self, encoder, decoder,latent_dim = 4096, args=None):
        super(Blackbox, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # opt and device
        self.opt = None
        self.device = get_device()
        self.dataset = args.dataset

        # MLP
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 42),
            torch.nn.ReLU(),
            torch.nn.Linear(42, 42),
        )
        self.last_hidden = torch.nn.Linear(42, args.num_Y)
        self.sigmoid = torch.nn.Sigmoid()
        self.int_C = args.num_C       
        
    def forward(self, x):

        # Image encoding        
        features  = self.encoder(x)[0]
        # MLP
        latents = self.mlp(features)
        # Prediction
        preds = self.last_hidden(latents)
        preds = self.sigmoid(preds)
        # Reconstruction
        recs = self.decoder(latents)[0]
        
        return {'LATENTS': latents, 'RECS': recs, 'PREDS': preds}
    
    def get_loss(self, args):
        if args.dataset in ['shapes3d', 'dsprites', 'kandinsky','mnist']:
            return Blackbox_Loss(args)
        else: 
            return NotImplementedError('Wrong dataset choice')
        
    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr)

    