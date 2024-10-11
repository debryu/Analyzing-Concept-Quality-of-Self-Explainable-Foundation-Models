import torch
import torch.nn.functional as F
from utils.args import *
from utils.conf import get_device
from utils.vae_loss import ConceptVAE_Loss

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Conceptual Variational Autoencoders.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class ConceptVae(torch.nn.Module):
    '''
    Conceptual Variational Autoencoder 
    from the paper https://arxiv.org/pdf/2203.11216.pdf
    '''
    NAME = 'conceptvae'
    def __init__(self, encoder= None, decoder=None,
                 n_images=1, c_split=(), args = None):      # Adding args as input
        super(ConceptVae,self).__init__()
        
        # bones of model
        self.encoder = encoder
        self.decoder = decoder
        
        # number of images, and how to split them in cat concepts
        self.n_images = n_images
        self.c_split = c_split 

        # opt and device
        self.opt = None
        self.device = get_device()

    def forward(self, x):
        # extract encodings
        cs, mus, logvars = [], [], []
        recs = []
        xs = torch.split( x, x.size(-1)//self.n_images, -1)
        for i in range(self.n_images):
            c, mu, logvar  = self.encoder(xs[i])
            #print(c.shape, mu.shape, logvar.shape)
            cs.append(c)
            mus.append(mu)
            logvars.append(logvar)
    
            #extract decodings

            # 1) convert to variational
            eps = torch.randn_like(logvar)
            zs = mu + eps*logvar.exp()

            index, hard_cs = 0, []
            #print(self.c_split)
            for cdim in self.c_split:
                hard_cs.append( F.gumbel_softmax(c[:, index:index+cdim] , tau=1, hard=False, dim=-1) )
                index += cdim
            #print(hard_cs)
            hard_cs = torch.cat(hard_cs, dim=-1)
            #print(zs.shape, hard_cs.shape)
            #print(hard_cs[0:3,:])
            # 2) pass to decoder
            decode = self.decoder(zs)
            recs.append(decode)
        
        # return everything
        cs = torch.cat(cs, dim=-1)
        mus = torch.cat(mus, dim=-1)
        logvars = torch.cat(logvars, dim=-1)
        recs = torch.cat(recs, dim=-1)
        #print(hard_cs)
        
        return {'RECS': recs, 'LATENTS': cs, 'MUS': mus, 'LOGVARS': logvars, 'GS':hard_cs} # replace latents with CS

    @staticmethod
    def get_loss(args):
        print(args)
        args.w_c = 0
        if args.dataset in ['shapes3d','mnist']:
            return ConceptVAE_Loss(args = args)
        else: return NotImplementedError('Wrong dataset choice')
        
    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr)