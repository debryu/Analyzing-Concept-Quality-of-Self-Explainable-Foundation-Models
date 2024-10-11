import torch
import numpy as np
import torch.nn.functional as F

def conditional_gen(model, mu_cluster, args):
    # select whether generate at random or not
    # if pC is None:
    #     pC = 5 * torch.randn((8, model.n_images, model.encoder.c_dim), device=model.device)
    #     # pC = torch.softmax(pC, dim=-1)

    # zs = torch.randn((8, model.n_images, model.encoder.latent_dim), device=model.device)

    # latents = []   
    # for _ in range(model.n_images):
    #     for i in range(len(model.c_split)):
    #         latents.append(zs[:,i,:])
    #         latents.append(F.gumbel_softmax(pC[:, i, :], tau=1, hard=True, dim=-1)) 
    
    latents = torch.randn((16, args.latent_dim), device=model.device)
    # Add the cluster mean, which is the same for every 16 images
    # mu_cluster shape = (42)
    # latents shape = (16, 42)
    # mu_cluster.unsqueeze(0).shape = (1, 42)
    # mu_cluster.unsqueeze(0).expand(16, -1).shape = (16, 42)
    #latents = latents + mu_cluster.unsqueeze(0).expand(16, -1)
    
    # generated images
    decode = model.decoder(latents).detach()

    return decode

def recon_visaulization(out_dict):
    images = out_dict['INPUTS'].detach()[:16]
    recons = out_dict['RECS'].detach()[:16]
    return torch.cat([images, recons], dim=0 )
