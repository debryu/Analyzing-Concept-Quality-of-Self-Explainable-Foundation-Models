import torch
from utils.args import *
from utils.conf import get_device
from utils.losses import Class_Match


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Learning via'
                                        'Concept Extractor .')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class CExt(torch.nn.Module):
    NAME = 'cext'
    def __init__(self, encoder, args): 
        super(CExt, self).__init__()
        
        # bones of the model 
        self.encoder = encoder

        # opt and device
        self.opt = None
        self.device = get_device()
        self.args = args

    def forward(self, x):
        cs =  self.encoder(x)[0]
        return {'LATENTS': cs}

    @staticmethod
    def get_loss(args):
        if args.dataset == 'shapes3d':
            return Class_Match(args, int_C=4)
        else:
            return NotImplementedError('Wrong Choice')
        
    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr)