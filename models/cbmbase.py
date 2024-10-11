import torch
from utils.args import *
from utils.conf import get_device
from models.cext import CExt
from utils.losses import CBM_Loss
from torch.nn import functional as F

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Learning via'
                                        'VAE-CSUP .')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class CBMbase(CExt):
    NAME = 'cbmbase'
    '''
    Vae with csup, for DIS datasets
    '''
    def __init__(self, encoder, decoder,args=None):
        super(CBMbase, self).__init__(encoder=encoder, args=args)

        # add projection layer
        self.linear = torch.nn.Linear(args.num_C, args.num_Y)
        self.int_C = args.num_C
        self.num_Y = args.num_Y
        # opt and device
        self.opt = None
        self.device = get_device()

    def forward(self, x):

        # Image encoding        
        concepts, out_dict = self.encoder(x)   # The previous "concepts, _ = self.encoder(x)" raised an error
        concepts = concepts[0]
        #print(concepts)
        preds = self.linear( torch.sigmoid(concepts))
        
        return {'LATENTS': concepts, 'PREDS': preds, 'ENCODER': out_dict['hidden_input']}
    
    def get_loss(self, args):
        if args.dataset in ['shapes3d', 'dsprites', 'kandinsky','mnist','celeba']:
            return CBM_Loss(args, int_C=args.num_C)
        else: 
            return NotImplementedError('Wrong dataset choice')
        
    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr)
