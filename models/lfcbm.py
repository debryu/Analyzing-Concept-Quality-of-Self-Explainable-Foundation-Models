import torch
from utils.conf import get_device
from utils.data_utils import get_target_model
import os
import json
import copy

class LFcbm(torch.nn.Module):
    NAME = 'lfcbm'
    def __init__(self, backbone_name, W_c, W_g, b_g, proj_mean, proj_std, device="cuda"):
        super().__init__()
        self.device = get_device()
        model, _ = get_target_model(backbone_name, device)
        #remove final fully connected layer
        if "clip" in backbone_name:
            self.backbone = model
        elif "cub" in backbone_name:
            self.backbone = lambda x: model.features(x)
        else:
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
            
        self.proj_layer = torch.nn.Linear(in_features=W_c.shape[1], out_features=W_c.shape[0], bias=False).to(device)
        self.proj_layer.load_state_dict({"weight":W_c})
            
        self.proj_mean = proj_mean
        self.proj_std = proj_std
        
        self.final = torch.nn.Linear(in_features = W_g.shape[1], out_features=W_g.shape[0]).to(device)
        self.final.load_state_dict({"weight":W_g, "bias":b_g})
        self.concepts = None
        
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.proj_layer(x)
        concepts = x
        proj_c = (x-self.proj_mean)/self.proj_std
        x = self.final(proj_c)
        #out_dict = {'concepts':concepts, 'normalized_concepts':proj_c, 'logits':x}
        return x, proj_c#, out_dict     # For now just proj_c is needed, so to make it work with the train code, I am returning only proj_c
    
    def get_loss(self, args):
        return NotImplementedError('No loss implemented')
        if args.dataset in ['shapes3d', 'dsprites', 'kandinsky','mnist']:
            return CBM_Loss(args, int_C=args.num_C)
        else: 
            return NotImplementedError('Wrong dataset choice')
        
    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr)

def load_cbm(load_dir, device):
    with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
        args = json.load(f)

    W_c = torch.load(os.path.join(load_dir ,"W_c.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir, "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir, "b_g.pt"), map_location=device)

    proj_mean = torch.load(os.path.join(load_dir, "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir, "proj_std.pt"), map_location=device)

    model = LFcbm(args['backbone'], W_c, W_g, b_g, proj_mean, proj_std, device)
    return model