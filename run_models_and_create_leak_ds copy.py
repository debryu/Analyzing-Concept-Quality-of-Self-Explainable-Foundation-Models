import os
import torch
from datasets import get_dataset
from models import get_model
from utils.utils import get_dataset_C_Y
import argparse
import importlib
from tqdm import tqdm
from metrics.completeness import completeness_havasi_mi
import numpy as np
from utils.utils import Leakage_Dataset
from sklearn.metrics import classification_report, accuracy_score
import math
''' 
    Run different model for collecting results
'''
def get_saved_models_to_run(saved_models_folder = '/data/ckpt/saved_models/evaluate/'):
  '''
    Get all the different model checkpoints to run from the models folder
  '''
  # The path where this file is stored
  path = os.path.dirname(os.path.realpath(__file__))
  # Folder where all the checkpoints are stored
  folder = path + saved_models_folder
  # Get a list of all folders inside the folder
  variations = os.listdir(folder)
  models = {}
  for v in variations:
      # Get all the different checkpoints (path)
      checkpoints = os.listdir(folder + v)
      c = {}
      for ckpt in checkpoints:
        c[ckpt] = folder + v + '/' + ckpt
      models[v] = c
  return models

def add_args(parser):
  parser.add_argument('--model', type=str,default='betaglancenet', help='Model for inference.')
  parser.add_argument('--dataset', type=str,default='shapes3d', help='Dataset')
  parser.add_argument('--split', type=str,default='test', help='Split')
  parser.add_argument('--latent_dim', type=int, default=42, help='Dimension of latent space')
  parser.add_argument('--c_sup', type=float, default=0,   help='Fraction of concept supervision on concepts')
  parser.add_argument('--which_c', type=int, nargs='+', default=[-1], help='Which concepts explicitly supervise (-1 means all)')
  # additional hyperparams
  parser.add_argument('--w_rec', type=float, default=1, help='Weight of Reconstruction')
  parser.add_argument('--beta',  type=float, default=2, help='Multiplier of KL')
  parser.add_argument('--z_capacity', type=float, default=2, help='Max capacity of KL')
  parser.add_argument('--w_c',   type=float, default=1, help='Weight of concept sup')
  
  # optimization params
  parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
  parser.add_argument('--warmup_steps', type=int, default=2, help='Warmup epochs.')
  parser.add_argument('--exp_decay', type=float, default=0.99, help='Exp decay of learning rate.')
  
  # learning hyperams
  parser.add_argument('--n_epochs',   type=int, default=50, help='Number of epochs per task.')
  parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
  return parser

def evaluate(model, loader, loss, version, beta, args, loc='test'):
    norm = len(loader)
    eval_losses = {}
    
    acc, cacc = 0, 0
    samples = 0
    # This is the boolean version of the concepts
    concepts_prediction = []
    concepts_gt = []

    all_labels = []
    all_labels_predictions = []
    all_concepts = []
    all_concepts_predictions = []
    all_images = []
    for i, data in enumerate(loader):
        images, labels, concepts = data 
        images, labels, concepts = images.to(model.device), labels.to(model.device), concepts.to(model.device)
      
        ''' CONTROL CONCEPTS SUPERVISION HERE '''
        # Mask the concepts
        # Only keep scale and shape, not color (orientation already masked in the dataset)
        #{'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 'scale': 8, 'shape': 4, 'orientation': 15}
        if version.split('_')[1] == 'partialconceptsupervision':
          concepts[:, :30] = -1
        if version.split('_')[1] == 'morepartialconceptsupervision':
          concepts[:, 30:] = -1
        if version.split('_')[1] == 'edgecase':
          if beta == 'beta1NoShape.pt':
            concepts[:, 38:] = -1
          elif beta == 'beta1OnlyUseful.pt':
            concepts[:, :20] = -1
          elif beta == 'beta1OnlyUseless.pt':
            concepts[:, 20:30] = -1 
            concepts[:, 38:] = -1

        out_dict = model(images)

        out_dict.update({'INPUTS': images, 'LABELS': labels, 'CONCEPTS': concepts})

        # collect the labels, concepts and images for the completeness metric
        all_labels.append(labels.detach().to('cpu').numpy())
        all_concepts.append(concepts.detach().to('cpu').numpy())
        var = out_dict['HIDDEN_INPUT'].detach().to('cpu')
        var = var.reshape(var.shape[0],-1).numpy()
        all_images.append(images.detach().to('cpu').numpy())

        batch_loss, losses = loss(out_dict, args)
        #print(f'batch_loss: {batch_loss}')
        #print(f'losses: {losses}')
        if i == 0:
            for key in losses.keys():
                eval_losses[loc+key] = losses[key] / norm
            eval_losses['loss'] = batch_loss.detach().item() / norm
        else:
            for key in losses.keys():
                eval_losses[loc+key] += losses[key] / norm
            eval_losses['loss'] += batch_loss.detach().item() / norm

        if args.model in ['betaplusglancenet']:
            reprs = out_dict['LOGITS']
        else: 
            reprs = out_dict['LATENTS']
        
        concepts = out_dict['CONCEPTS']
        reprs = reprs[:, : concepts.size(-1)]

        #print(f'reprs: {reprs[0]}')
        c_predictions = torch.sigmoid(reprs)
        all_concepts_predictions.append(c_predictions.detach().cpu())
        all_labels_predictions.append(out_dict['PREDS'].detach().cpu())
        #print(f'sigm-reprs: {c_predictions[0]}')
        # assign 0 to the concept if it is less than 0.5, 1 otherwise
        c_predictions = (c_predictions > 0.5).type(torch.float)
        #print(f'bool-reprs: {c_predictions[0]}')
        #print(f'gt: {concepts[0]}')
        #print(f'correct: {(c_predictions[0] == concepts[0]).sum().item()}')
        #print('c_predictions:', c_predictions.shape)
        # Count the number of masked concepts
        masked_concepts = (concepts == -1).sum().item()
        #print(f'masked_concepts: {masked_concepts}')
        total_concepts = concepts.shape[0] * concepts.shape[1] - masked_concepts
        cacc += (c_predictions == concepts).sum().item() / total_concepts
        
        if args.model in ['cvae','conceptvae','betavae']:
            pass
        else:
            label_predictions = out_dict['PREDS'].argmax(dim=1)
            total_predictions = labels.shape[0]
            acc += (label_predictions == labels).sum().item() / total_predictions
        samples += labels.size(0)
        concepts_prediction.append(c_predictions.detach().cpu())
        concepts_gt.append(out_dict['CONCEPTS'].cpu())

    
    # Store all the predictions and concepts (not a single batch)
    #Compute the DCI disentanglement metric

    cacc, acc = cacc / norm, acc / norm
    dci_ingredients = (torch.cat(concepts_prediction, dim = 0), torch.cat(concepts_gt, dim = 0))
    
    all_labels = np.concatenate(all_labels)
    all_labels_predictions = np.concatenate(all_labels_predictions)
    all_concepts = np.concatenate(all_concepts)
    all_concepts_predictions = np.concatenate(all_concepts_predictions)
    all_images = np.concatenate(all_images)

    stats_dict = {}
    stats_dict['labels'] = all_labels
    stats_dict['labels_predictions'] = all_labels_predictions
    stats_dict['concepts'] = all_concepts
    stats_dict['concepts_predictions'] = all_concepts_predictions
    stats_dict['images'] = all_images
    stats_dict['dci_ingredients'] = dci_ingredients
    return eval_losses, cacc, acc, stats_dict

def evaluate_models(model,dl,args):
  '''
    Evaluate the model
  '''
  output = {}

  checkpoints = get_saved_models_to_run()
  for version in checkpoints:
    output[version] = {}
    for ckpt in checkpoints[version]:
      
      path = checkpoints[version][ckpt]
      loss  = model.get_loss(args)
      model.load_state_dict(torch.load(path))
      model.to(model.device)
      print('--->    Chosen device: ', model.device, '\n')
      ver = version.split('_')[1]
      beta = ckpt.split('_')[3]
      seed = ckpt.split('_')[2]

      # Seed 10 = cycle annealing with 32 z_capacity
      # Seed 12 = cycle annealing with 64 z_capacity
      # Seed 128 = cycle annealing with 128 z_capacity
      # Seed 14 = beta-vae no concept supervision
      if seed in ['seed14']:
        output[version][ckpt] = {}
        _, concept_acc, label_acc, stats_dict = evaluate(model, dl, loss, version, beta, args, args.split)

        concepts = torch.tensor(stats_dict['concepts_predictions'])
        labels = torch.tensor(stats_dict['labels'])
        # Compute dataset frequencies for class 0 and 1
        classes = labels.unique()
        frequency = (labels == classes[1]).sum().item() / len(labels)
        #print(f'Frequency of class 1: {frequency}')

        train_conc = concepts[:20000,:]
        train_labels = labels[:20000]
        test_conc = concepts[20000:,:]
        test_labels = labels[20000:]
        # Save the leakage dataset
        train_ds = Leakage_Dataset(train_conc, train_labels)
        test_ds = Leakage_Dataset(test_conc, test_labels)
        leakage_dataset = { 'train': train_ds, 'test': test_ds}
        # Save the leakage dataset
        torch.save(leakage_dataset, os.path.dirname(os.path.realpath(__file__)) + f'/data/leakage_dataset/LD_{ver}_{seed}_{beta}.pth')

        print(f'Ver: {ver} - Beta: {beta} - Concept Accuracy: {concept_acc} - Label Accuracy: {label_acc}')
        # Run the a full accuracy-precision-recall-f1 report
        concepts_pred_list = (stats_dict['concepts_predictions'] > 0.5).astype(int).tolist()
        concepts_gt_list = stats_dict['concepts'].astype(int).tolist()
        # Reformat all the concepts as a single list (and not as a list of list)
        c_p_l = []
        c_gt_l = []
        for i, sample in enumerate(concepts_gt_list):
           for j,c in enumerate(sample):
             if c == -1:
                # Ignore masked concepts
                continue
             else:
                # shift every value by 2 times the index of the concept
                # This way we can have a unique identifier for each concept
                # And we can compute the classification report for each concept
                c_gt_l.append(c + j*2) 
                c_p_l.append(concepts_pred_list[i][j] + j*2) 

        concepts_pred_list = c_p_l
        concepts_gt_list = c_gt_l
        labels_pred_list = stats_dict['labels_predictions'].argmax(axis = 1).tolist()
        labels_gt_list = stats_dict['labels'].tolist()
        
        concepts_report = classification_report(concepts_gt_list, concepts_pred_list, output_dict=True)
        labels_report = classification_report(labels_gt_list, labels_pred_list, target_names = ['label_0', 'label_1'], output_dict=True)
        print('Concepts macro avg precision: ', concepts_report['macro avg']['precision'])
        print('Labels macro avg precision: ', labels_report['macro avg']['precision'])
        output 
        # Compute the completeness metric
        cs_havasi_mi = completeness_havasi_mi(stats_dict['labels'], stats_dict['concepts'], stats_dict['images'])
        print('Completeness score Havasi-MI: ', cs_havasi_mi)

        output[version][ckpt]['concept_accuracy'] = concept_acc
        output[version][ckpt]['label_accuracy'] = label_acc
        output[version][ckpt]['concept_report'] = concepts_report
        output[version][ckpt]['label_report'] = labels_report
        output[version][ckpt]['completeness_score'] = cs_havasi_mi
        output[version][ckpt]['concepts_macroavgprecision'] = concepts_report['macro avg']['precision']
        output[version][ckpt]['labels_macroavgprecision'] = labels_report['macro avg']['precision']
      else:
        print('Skipping seed: ', seed)
  return output
      
        

  
parser = argparse.ArgumentParser(description='Run and Evaluate models', allow_abbrev=False)

torch.set_num_threads(4)
parser = add_args(parser)
# Get the arguments
args = parser.parse_args()
args.num_C, args.num_Y = get_dataset_C_Y(args)
# Import the dataset
dataset = get_dataset(args)
# Import the model
mod = importlib.import_module('models.' + args.model)
encoder, decoder  = dataset.get_backbone(args)
model = get_model(args, encoder, decoder) 
# Import the dataloaders
train_loader, val_loader, (test_loader, ood_loader) = dataset.get_data_loaders()
if args.split == 'val':
  data_loader = val_loader
elif args.split == 'ood':
  data_loader = ood_loader
else:
  data_loader = test_loader

res = evaluate_models(model,data_loader,args)
versions = res.keys()
for version in versions:
   checkpoints = res[version].keys()
   for check in checkpoints:
      print('______________________________________________________________________________________')
      print(f'Version: {version} - Checkpoint: {check}')
      print(f'Concept Accuracy: {res[version][check]["concept_accuracy"]}')
      print(f'Concepts Macro Avg Precision: {res[version][check]["concepts_macroavgprecision"]}')
      print(f'Label Accuracy: {res[version][check]["label_accuracy"]}')
      print(f'Labels Macro Avg Precision: {res[version][check]["labels_macroavgprecision"]}')
      print(f'Completeness Score: {res[version][check]["completeness_score"]}')
      print('\n')






