import torch
import torch.nn.functional as F
from utils.args import *
from utils.conf import get_device
from utils.vae_loss import betaVAE_Loss

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Beta Variational-Conceptual Autoencoders.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class BETAVae(torch.nn.Module):
    NAME = 'betavae'
    def __init__(self, encoder= None, decoder=None,
                 n_images=1, c_split=(), args = None):      # Adding args as input
        super(BETAVae,self).__init__()
        
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
        mus, logvars, recs = [], [], []
        latents = []
        xs = torch.split( x, x.size(-1)//self.n_images, -1)
        for i in range(self.n_images):
            mu, logvar  = self.encoder(xs[i])
            #print('mu', mu.mean(), mu.min(), mu.max())
            #print('logvar', logvar.mean(), logvar.min(), logvar.max())
            mus.append(mu)
            logvars.append(logvar)

            # 1) Reparameterize
            eps = torch.randn_like(logvar)
            zs = mu + eps*logvar.exp()

            decode = self.decoder(zs)
            recs.append(decode)
            L = len(eps)
            latents.append((mu + eps*logvar.exp()).view(L, -1))
        # return everything
        latents = torch.cat(latents, dim=-1)
        mus = torch.cat(mus, dim=-1)
        logvars = torch.cat(logvars, dim=-1)
        recs = torch.cat(recs, dim=-1)
        return {'RECS': recs, 'MUS': mus, 'LOGVARS': logvars, 'LATENTS':latents} # replace latents with CS

    @staticmethod
    def get_loss(args):
        print(args)
        args.w_c = 0
        if args.dataset in ['shapes3d','mnist']:
            return betaVAE_Loss(args = args)
        else: return NotImplementedError('Wrong dataset choice')
        
    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr)