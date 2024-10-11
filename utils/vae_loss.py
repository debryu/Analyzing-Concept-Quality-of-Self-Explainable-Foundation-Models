import torch.nn.functional as F
import torch

def betaVAE_ELBO(out_dict: dict, args, Burgess = True):
    recs, inputs, mus, logvars  = out_dict['RECS'], out_dict['INPUTS'], out_dict['MUS'], out_dict['LOGVARS']

    L = len(recs)
    #print(inputs.size(), recs.size())
    assert inputs.size() == recs.size(), f'{len(inputs)}-{len(recs)}'
    #print('recs',recs.min(), recs.max())
    #print(inputs.min(), inputs.max())
    recon = F.mse_loss(recs, inputs, reduction='mean')
    #print(mus.size(), logvars.size())
    
    using_mu_cluster = False
    if 'mu_cluster' in out_dict.keys():
        if out_dict['mu_cluster'] is not None:
            mus = mus - out_dict['mu_cluster']
            using_mu_cluster = True
    #else:
        #print("\n!!! Not using mu_cluster !!!\n")
    if Burgess and using_mu_cluster:
        kld   =  (-0.5 * (1 + logvars - mus ** 2 - logvars.exp()).sum(1) - args.z_capacity).abs().mean()
    elif Burgess and not using_mu_cluster:
        kld   =  (-0.5 * (1 + logvars - logvars.exp()).sum(1) - args.z_capacity).abs().mean()
        
    else:
        kld = (-0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp())).mean()
    
    #print(kld)
    losses = {'recon-loss': recon.item(), 'kld':kld.item()}
    
    return recon + args.beta * kld, losses 

def conceptVAE_ELBO(out_dict: dict, args):
    recs, inputs, concepts, gs, labels  = out_dict['RECS'], out_dict['INPUTS'], out_dict['CONCEPTS'], out_dict['GS'], out_dict['LABELS']   
    L = len(recs)
    
    assert inputs.size() == recs.size(), f'{len(inputs)}-{len(recs)}'
    #print('recs',recs.min(), recs.max())
    #print(inputs.min(), inputs.max())
    recon = F.mse_loss(recs, inputs, reduction='mean')
    #print(mus.size(), logvars.size())
    kld = torch.nn.functional.binary_cross_entropy(gs, concepts.float(), reduction='mean')
    #print(kld)
    losses = {'recon-loss': recon.item(), 'kld':kld.item()}
    
    return recon + args.beta * kld, losses 

def VAE_Concept_Match(out_dict: dict, args=None, usedC=6):
    if args.model == 'betaplusglancenet':
        reprs = out_dict['LOGITS']
    else: 
        reprs = out_dict['LATENTS']

    concepts = out_dict['CONCEPTS'].to(torch.float)
    reprs = reprs[:, : concepts.size(-1)]
    mask = (concepts[0,:] != -1)
    if 'CE_WEIGHT' in out_dict.keys():
        if out_dict['CE_WEIGHT'] is not None:
            weights = out_dict['CE_WEIGHT']
        else:
            weights = torch.ones_like(mask)
    else:
        weights = torch.ones_like(mask)
    loss = torch.zeros((), device=reprs.device)
    #print(usedC)
    if mask.sum() > 0:
        # for i in range(concepts.size(-1)):
        loss += torch.nn.functional.binary_cross_entropy(
            torch.sigmoid(reprs[:,mask][:,:usedC] ), # sigmoid (reprs[:, i])
            (concepts[:,mask][:,:usedC]), # sigmoid (concepts[:,i])
            reduction='mean',
            weight = weights[mask][:usedC]
            )
        
    losses = {'c-loss': loss.item()}

    return loss, losses

def single_label_loss(out_dict: dict):
    preds, ys  = out_dict['PREDS'], out_dict['LABELS']

    # print(preds[:3])
    # print(ys[:3])
    if 'CE_weight_labels' in out_dict.keys():
        if out_dict['CE_weight_labels'] is not None:
            weights = out_dict['CE_weight_labels']
        else:
            weights = torch.ones(2).to(ys.device)
    else:
        weights = torch.ones(2).to(ys.device)
    ys = ys.type(torch.long)
    pred_loss = F.cross_entropy(preds, ys.view(-1), reduction='mean', weight=weights)
    
    losses = {'pred-loss': pred_loss.item()}
    
    return pred_loss, losses 

class betaVAE_Loss(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def forward(self, out_dict, args):
        loss1, losses1 = betaVAE_ELBO(out_dict, self.args)  
        if args.w_c > 0:
            loss2, losses2 = VAE_Concept_Match(out_dict, self.args)
            losses1.update(losses2)   
            return loss1+ args.w_c * loss2, losses1 
        else:
            return loss1, losses1
        
class CVAE_Loss(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def forward(self, out_dict, args):
        loss1, losses1 = betaVAE_ELBO(out_dict, self.args, Burgess=False)  
        if args.w_c > 0:
            loss2, losses2 = VAE_Concept_Match(out_dict, self.args)
            losses1.update(losses2)   
            return loss1+ args.w_c * loss2, losses1 
        else:
            return loss1, losses1
        
class ConceptVAE_Loss(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def forward(self, out_dict, args):
        loss1, losses1 = conceptVAE_ELBO(out_dict, self.args)  
        if args.w_c > 0:
            loss2, losses2 = VAE_Concept_Match(out_dict, self.args)
            losses1.update(losses2)   
            return loss1+ args.w_c * loss2, losses1 
        else:
            return loss1, losses1

    
class betaGlanceNet_Loss(torch.nn.Module):
    def __init__(self, args, int_C=6) -> None:
        super().__init__()
        self.args = args
        self.int_C = int_C

    def forward(self, out_dict, args):
        loss1, losses1 = betaVAE_ELBO(out_dict, self.args)        
        loss2, losses2 = VAE_Concept_Match(out_dict, args, usedC=self.int_C)
        loss3, losses3 = single_label_loss(out_dict)

        losses1.update(losses2)
        losses1.update(losses3)

        return args.w_label*loss3 + args.w_rec*loss1+ args.w_c * loss2, losses1