import torch
from utils.args import *
from utils.conf import get_device
from models.vaesup import VAESUP
from utils.vae_loss import betaGlanceNet_Loss

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Learning via'
                                        'VAE-CSUP .')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class betaplusGlanceNet(VAESUP):
    NAME = 'betaplusglancenet'
    '''
    Vae with csup, for DIS datasets
    '''
    def __init__(self, encoder, decoder, args=None):
        super(betaplusGlanceNet, self).__init__(encoder=encoder, decoder=decoder, args=args)

        # opt and device
        self.opt = None
        self.device = get_device()

        # define projection layer
        if args.dataset == 'kandinsky':
            self.int_C = 6
            self.projection = torch.nn.Linear(self.int_C*2, self.int_C)
            self.linear = torch.nn.Linear(self.int_C, 2)
        else:
            return NotImplementedError('wrong choice of dataset')
        
    
    def forward(self, x):

        # Image encoding        
        mu, logvar  = self.encoder(x)

        # 1) add variational vars

        eps = torch.randn_like(logvar)
        L = len(eps)
        
        latents = (mu + eps*logvar.exp()).view(L, -1)

        # 2) pass to decoder
        recs = self.decoder(latents)

        # 3) project last-layer
        concepts = self.projection(latents[:, :self.int_C*2])
        preds = self.linear( torch.sigmoid(concepts) )
        
        return {'MUS': mu, 'LOGVARS': logvar, 'LATENTS': latents, 'RECS': recs, 'PREDS': preds, 'LOGITS':concepts}
    
    def get_loss(self, args):
        if args.dataset in ['shapes3d', 'dsprites', 'kandinsky']:
            return betaGlanceNet_Loss(int_C=6, args=args)
        else: 
            return NotImplementedError('Wrong dataset choice')
        
    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr)