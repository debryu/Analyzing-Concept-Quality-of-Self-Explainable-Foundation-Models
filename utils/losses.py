import torch.nn.functional as F
import torch

def Concept_Match(out_dict: dict, args=None, usedC=6):
    if args.model == 'betaplusglancenet':
        reprs = out_dict['LOGITS']
    else: 
        reprs = out_dict['LATENTS']

    #concepts = out_dict['CONCEPTS'].to(torch.float)
    #concepts = out_dict['CONCEPTS']
    #concepts = concepts.to(torch.float32)
    concepts = out_dict['CONCEPTS'].to(torch.float)
    reprs = reprs[:, : concepts.size(-1)]
    mask = (concepts[0,:] != -1)
    
    loss = torch.zeros(()).to(reprs.device)
    if mask.sum() > 0:
        # for i in range(concepts.size(-1)):
        loss += torch.nn.functional.binary_cross_entropy(
            torch.sigmoid(reprs[:,mask][:,:usedC] ), # sigmoid (reprs[:, i])
            (concepts[:,mask][:,:usedC]), # sigmoid (concepts[:,i])
            reduction='mean'
            )
        
    losses = {'c-loss': loss.item()}

    return loss, losses



def single_label_loss(out_dict: dict):
    preds, ys  = out_dict['PREDS'], out_dict['LABELS']
    preds = preds.type(torch.FloatTensor).to(ys.device)
    ys = ys.type(torch.LongTensor).to(preds.device)
    # Weights are inversely proportional to the class frequency
    W = torch.tensor([1.038,27.3]).to(preds.device) # Weights for Shapes3d
    W = torch.tensor([1.0,1.0]).to(preds.device)    # Set equal weights
    pred_loss = F.cross_entropy(preds, ys.view(-1), reduction='mean', weight = W)
    losses = {'pred-loss': pred_loss.item()}
    
    return pred_loss, losses 
    
class CBM_Loss(torch.nn.Module):
    def __init__(self, args, int_C=6) -> None:
        super().__init__()
        self.args = args
        self.int_C = int_C

    def forward(self, out_dict, args):
        loss1, losses1 = Concept_Match(out_dict, args, usedC=self.int_C)
        loss2, losses2 = single_label_loss(out_dict)

        losses1.update(losses2)
        
        return args.w_label*loss2 + args.w_c*loss1, losses1    # Temporary

class Class_Match(torch.nn.Module):
    def __init__(self, args, int_C=6) -> None:
        super().__init__()
        self.args = args
        self.int_C = int_C
    
    def forward(self, out_dict, args=None):
        if self.args.model == 'betaplusglancenet':
            reprs = out_dict['LOGITS']
        # elif self.args.model == 'betaplusglancenet':
        #     reprs = out_dict['Cs']
        else: 
            reprs = out_dict['LATENTS']

        concepts = out_dict['CONCEPTS'].to(torch.long)
        mask = (concepts != -1)
        
        loss = torch.zeros((), device=reprs.device)
        if mask.sum() > 0:
            # for i in range(concepts.size(-1)):
            loss += torch.nn.functional.cross_entropy(
                reprs[mask][:,:self.int_C] , # sigmoid (reprs[:, i])
                concepts[mask], # sigmoid (concepts[:,i])
                reduction='mean'
                )
            
        losses = {'c-loss': loss.item()}

        return loss, losses
    
class Blackbox_Loss(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def forward(self, out_dict, args):
        #recs = out_dict['RECS']
        #labels = out_dict['LABELS']
        #latents = out_dict['LATENTS']
        preds, ys  = out_dict['PREDS'], out_dict['LABELS']
        preds = preds.type(torch.FloatTensor).to(ys.device)
        ys = ys.type(torch.LongTensor).to(preds.device)
        pred_loss = F.cross_entropy(preds, ys.view(-1), reduction='mean')
        
        losses = {'pred-loss': pred_loss.item()}
        #TODO add more losses like reconstruction loss

        return pred_loss, losses