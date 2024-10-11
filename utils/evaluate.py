import torch
import numpy as np
import wandb

from torchvision.utils import make_grid
from utils.wandb_logger import *
from utils.status import progress_bar
from datasets.utils.base_dataset import BaseDataset
from utils.generative import conditional_gen, recon_visaulization
from metrics.dci_framework import _compute_dci
from utils.checkpoint import CheckpointSaver
from utils.log_images import *
from metrics.evaluate_val_test import evaluate_metrics
from utils.osr import run_osr
from tqdm import tqdm

from warmup_scheduler import GradualWarmupScheduler

def eval(model, dataset: BaseDataset, _loss, args, thr_r = None, thr_y = None, prot = None, plot = False):
  # saving location
  ckpt_saver = CheckpointSaver(args.checkout, decreasing=False, args=args)

  # Default Setting for Training
  model.to(model.device)
  train_loader, test_loader, ood_loader = dataset.get_data_loaders()
  print('\n--- Start of Evaluation ---\n')
  # Evaluate performances on val or test
  norm = len(test_loader)
  eval_losses = {}
  accuracies = {}
  acc, cacc = 0, 0
  samples = 0
  loc = 'test'
  for i, data in enumerate(test_loader):
    images, labels, concepts = data 
    images, labels, concepts = images.to(model.device), labels.to(model.device), concepts.to(model.device)

    out_dict = model(images)
    out_dict.update({'INPUTS': images, 'LABELS': labels, 'CONCEPTS': concepts})
    if prot is not None:
      if i == 0:
        mask = run_osr(out_dict['RECS'], out_dict['INPUTS'], out_dict['MUS'], thr_r, thr_y, prot, plot=plot)
      else:
        mask = run_osr(out_dict['RECS'], out_dict['INPUTS'], out_dict['MUS'], thr_r, thr_y, prot, plot=False)
      # {'MUS': mu, 'LOGVARS': logvar, 'LATENTS': latents, 'RECS': recs, 'PREDS': preds}
      out_dict['MUS'] = out_dict['MUS'][mask]
      out_dict['LOGVARS'] = out_dict['LOGVARS'][mask]
      out_dict['LATENTS'] = out_dict['LATENTS'][mask]
      out_dict['RECS'] = out_dict['RECS'][mask]
      out_dict['PREDS'] = out_dict['PREDS'][mask]
      out_dict['INPUTS'] = out_dict['INPUTS'][mask]
      out_dict['LABELS'] = out_dict['LABELS'][mask]
      out_dict['CONCEPTS'] = out_dict['CONCEPTS'][mask]
      images = images[mask]
      labels = labels[mask]
      concepts = concepts[mask]
    
    _, losses = _loss(out_dict, args)

    if i == 0:
        for key in losses.keys():
            eval_losses[loc+key] = losses[key] / norm
    else:
        for key in losses.keys():
            eval_losses[loc+key] += losses[key] / norm

    if args.model == 'betaplusglancenet':
        reprs = out_dict['LOGITS']
    else: 
        reprs = out_dict['LATENTS']
    
    concepts = out_dict['CONCEPTS']
    reprs = reprs[:, : concepts.size(-1)]
    c_predictions = torch.sigmoid(reprs)
    # assign 0 to the concept if it is less than 0.5, 1 otherwise
    c_predictions = (c_predictions > 0.5).type(torch.int64)
    total_concepts = concepts.shape[0] * concepts.shape[1]
    if total_concepts != 0:
      cacc += (c_predictions == concepts).sum().item() / total_concepts
    
    if args.model in ['cvae','conceptvae']:
        pass
    else:
        label_predictions = out_dict['PREDS'].argmax(dim=1)
        total_predictions = labels.shape[0]
        if total_predictions != 0:
          acc += (label_predictions == labels).sum().item() / total_predictions
    samples += labels.size(0)

  cacc, acc = cacc / norm, acc / norm
  accuracies[loc+'-acc'] = acc
  accuracies[loc+'-cacc'] = cacc
  # OOD
  acc, cacc = 0, 0
  samples = 0
  loc = 'ood'
  norm = len(ood_loader)
  for i, data in enumerate(tqdm(ood_loader)):
    images, labels, concepts = data 
    images, labels, concepts = images.to(model.device), labels.to(model.device), concepts.to(model.device)

    out_dict = model(images)

    out_dict.update({'INPUTS': images, 'LABELS': labels, 'CONCEPTS': concepts})
    if prot is not None:
      if i == 0:
        mask = run_osr(out_dict['RECS'], out_dict['INPUTS'], out_dict['MUS'], thr_r, thr_y, prot, plot=plot)
      else:
        mask = run_osr(out_dict['RECS'], out_dict['INPUTS'], out_dict['MUS'], thr_r, thr_y, prot, plot=False)
      # {'MUS': mu, 'LOGVARS': logvar, 'LATENTS': latents, 'RECS': recs, 'PREDS': preds}
      out_dict['MUS'] = out_dict['MUS'][mask]
      out_dict['LOGVARS'] = out_dict['LOGVARS'][mask]
      out_dict['LATENTS'] = out_dict['LATENTS'][mask]
      out_dict['RECS'] = out_dict['RECS'][mask]
      out_dict['PREDS'] = out_dict['PREDS'][mask]
      out_dict['INPUTS'] = out_dict['INPUTS'][mask]
      out_dict['LABELS'] = out_dict['LABELS'][mask]
      out_dict['CONCEPTS'] = out_dict['CONCEPTS'][mask]
      images = images[mask]
      labels = labels[mask]
      concepts = concepts[mask]

    _, losses = _loss(out_dict, args)

    if i == 0:
        for key in losses.keys():
            eval_losses[loc+key] = losses[key] / norm
    else:
        for key in losses.keys():
            eval_losses[loc+key] += losses[key] / norm

    if args.model == 'betaplusglancenet':
        reprs = out_dict['LOGITS']
    else: 
        reprs = out_dict['LATENTS']
    
    concepts = out_dict['CONCEPTS']
    reprs = reprs[:, : concepts.size(-1)]
    c_predictions = torch.sigmoid(reprs)
    # assign 0 to the concept if it is less than 0.5, 1 otherwise
    c_predictions = (c_predictions > 0.5).type(torch.int64)
    total_concepts = concepts.shape[0] * concepts.shape[1]
    if total_concepts != 0:
      cacc += (c_predictions == concepts).sum().item() / total_concepts
    
    if args.model in ['cvae','conceptvae']:
        pass
    else:
        label_predictions = out_dict['PREDS'].argmax(dim=1)
        total_predictions = labels.shape[0]
        if total_predictions != 0:
          acc += (label_predictions == labels).sum().item() / total_predictions
    samples += labels.size(0)

  cacc, acc = cacc / norm, acc / norm
  accuracies[loc+'-acc'] = acc
  accuracies[loc+'-cacc'] = cacc

  return eval_losses, accuracies  