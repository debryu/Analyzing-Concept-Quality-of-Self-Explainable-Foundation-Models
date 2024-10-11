import torch
from utils.args import *
from utils.conf import get_device
from models.vaesup import VAESUP
from utils.vae_loss import betaGlanceNet_Loss
import torch.nn.functional as F

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Learning via'
                                        'VAE-CSUP .')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class betaGlanceNet(VAESUP):
    NAME = 'betaglancenet'
    '''
    Vae with csup, for DIS datasets
    '''
    def __init__(self, encoder, decoder, args=None):
        super(betaGlanceNet, self).__init__(encoder=encoder, decoder=decoder, args=args)

        # opt and device
        self.opt = None
        self.device = get_device()
        self.dataset = args.dataset
        self.int_C = args.num_C
        self.num_Y = args.num_Y
        # define projection layer
        self.linear = torch.nn.Linear(args.num_C, args.num_Y) 
        self.z_from_y = torch.nn.Linear(args.num_Y, args.num_C)

        self.linear30 = torch.nn.Linear(30, args.num_Y)  # For partial concept supervision
        self.linear12 = torch.nn.Linear(12, args.num_Y)  # For partial concept supervision
        self.linear6 = torch.nn.Linear(6, args.num_Y)  # For partial concept supervision
        self.linear3 = torch.nn.Linear(3, args.num_Y)  # For partial concept supervision
        self.linear2 = torch.nn.Linear(2, args.num_Y)     
        
    def forward(self, x):

        # Image encoding        
        (mu, logvar), out_dict  = self.encoder(x)       # Added the out dict to get the hidden input, used for the completeness metric of havasi paper 

        # 1) add variational vars
        eps = torch.randn_like(logvar)
        L = len(eps)
        latents = (mu + eps*logvar.exp()).view(L, -1)

        # 2) pass to decoder
        recs = self.decoder(latents)

        # 3) project last-layer
        preds = self.linear(torch.sigmoid(latents)) 
        #preds = self.linear12(torch.sigmoid(latents[:, 30:self.int_C])) # Temporary
        
        return {'MUS': mu, 'LOGVARS': logvar, 'LATENTS': latents, 'RECS': recs, 'PREDS': preds, 'ENCODER': out_dict['hidden_input']}
    
    def get_loss(self, args):
        if args.dataset in ['shapes3d', 'dsprites', 'kandinsky','mnist','celeba']:
            return betaGlanceNet_Loss(int_C=args.num_C, args=args)
        else: 
            return NotImplementedError('Wrong dataset choice')
        
    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr)

    def compute_prior(self, labels):
        labels = labels.to(dtype=torch.long, device=self.device)   
        one_hot = F.one_hot(labels, num_classes=self.num_Y).to(dtype=torch.float, device=self.device)
        return self.z_from_y(one_hot)

    