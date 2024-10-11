import torch
import os
from utils.conf import create_path 
import logging
import wandb
import numpy as np
from datetime import datetime
import pickle

def create_load_ckpt(model, args):
    create_path('data/runs')
    
    PATH = f'data/runs/{args.dataset}-{args.model}-{args.latent_dim}-{args.num_C}-seed{args.seed}-start224_s.pt'
    
    if args.checkin is not None:
        model.load_state_dict(torch.load(args.checkin))
    
    elif os.path.exists(PATH):
        print('Loaded',PATH, '\n')
        model.load_state_dict(torch.load(PATH))
    else:
        print('Created',PATH, '\n')
        torch.save(model.state_dict(), PATH)

    return model        

class CheckpointSaver:
    def __init__(self, dirpath, decreasing=True, top_n=5, args=None):
        """
        dirpath: Directory path where to store all model weights 
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n 
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf
        self.wandb = args.wandb
        
    def __call__(self, model, epoch, metric_val, args):

        model_type = model.__class__.__name__
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #check if the folder exists
        model_save_folder = self.dirpath + f'/{model_type}_epoch{epoch}_seed{args.seed}_beta{args.beta}_{current_time}'
        if not os.path.exists(model_save_folder):
            os.makedirs(model_save_folder)
        model_path = os.path.join(model_save_folder, model_type + f'{args.dataset}_epoch{epoch}_seed{args.seed}_beta{args.beta}.pt')
        # Use the same loss used in training
        metric_val = metric_val['loss']
        #metric_val = metric_val[list(metric_val.keys())[0]]
        #print(self.decreasing)
        save = metric_val<self.best_metric_val if self.decreasing else metric_val>self.best_metric_val
        #print(metric_val, self.best_metric_val, save)
        
        if save: 
            logging.info(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}, & logging model weights to W&B.")
            self.best_metric_val = metric_val
            torch.save(model.state_dict(), model_path)
            pickle.dump(args, open(model_save_folder+'/args.pkl', 'wb'))
            with open(model_save_folder+f'/{args.sup_version}.txt', 'w') as f:
                f.write(str(args.dataset))
            self.log_artifact(f'model-ckpt-epoch-{epoch}.pt', model_path, metric_val)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)
        else: 
            print(f' max:{self.best_metric_val} !!! Not saving the model !!!')
            return 'failed'
        if len(self.top_model_paths)>self.top_n: 
            self.cleanup()
        
    
    def log_artifact(self, filename, model_path, metric_val):
        if self.wandb is not None:
            artifact = wandb.Artifact(filename, type='model', metadata={'Validation score': metric_val})
            artifact.add_file(model_path)
            wandb.run.log_artifact(artifact)        
    
    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]