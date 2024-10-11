import torch
from utils.args import *
from utils.conf import get_device
from models.cext import CExt
from utils.vae_loss import betaVAE_Loss

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Learning via'
                                        'VAE-CSUP .')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class VAESUP(CExt):
    NAME = 'vaesup'
    '''
    Vae with csup, for DIS datasets
    '''
    def __init__(self, encoder, decoder, args=None):
        super(VAESUP, self).__init__(encoder=encoder, args=args)
        self.decoder = decoder

        # opt and device
        self.opt = None
        self.device = get_device()

    def forward(self, x):

        # Image encoding        
        mu, logvar  = self.encoder(x)

        # 1) add variational vars

        eps = torch.randn_like(logvar)
        L = len(eps)
        
        latents = (mu + eps*logvar.exp()).view(L, -1)

        # 2) pass to decoder
        recs = self.decoder(latents)
        
        return {'MUS': mu, 'LOGVARS': logvar, 'LATENTS': latents, 'RECS': recs}
    
    def get_loss(self, args):
        if args.dataset in ['shapes3d', 'dsprites', 'kandinsky']:
            return betaVAE_Loss(args)
        else: 
            return NotImplementedError('Wrong dataset choice')
        
    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr)