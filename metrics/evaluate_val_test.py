import torch
from metrics.dci_framework import _compute_dci
import os
import pickle
from torch import nn
VAE_MODELS = ['betaglancenet','betavae','conceptvae','cvae']

def evaluate_metrics(model, loader, loss, args, loc='test', use_prior = False, prior_weight = 100):
    norm = len(loader)
    eval_losses = {}

    acc, cacc = 0, 0
    samples = 0
    concepts_prediction = []
    concepts_gt = []

    CE_weight = None
    CE_weight_labels = None
    if os.path.exists(f'data/ds_freqs/{args.dataset}_freq.pkl'):
            ds_freq = pickle.load(open(f'data/ds_freqs/{args.dataset}_freq.pkl', 'rb'))
            CE_weight = ds_freq['CE_weight']
            frequencies = ds_freq['frequencies']
            CE_weight_labels = ds_freq['CE_weight_labels']
    else:
        raise ValueError('No dataset frequencies found')

    for i, data in enumerate(loader):
        
        images, labels, concepts = data 
        images, labels, concepts = images.to(model.device), labels.to(model.device), concepts.to(model.device)
        # Mask the concepts
        # Only keep scale and shape, not color (orientation already masked in the dataset)
        #{'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 'scale': 8, 'shape': 4, 'orientation': 15}
        
        ''' CONTROL CONCEPTS SUPERVISION HERE '''
        #concepts[:, :30] = -1   # Supervision on 12 concepts
        #concepts[:, 30:] = -1   # Supervision on 30 concepts
        #concepts[:, 38:] = -1   # Supervision on everything but shape
        #concepts[:, :20] = -1  # Supervision on useful concepts

        #concepts[:, 20:30] = -1
        #concepts[:, 38:] = -1   # Supervision on useless concepts

        #concepts[:, 20] = -1
        #concepts[:, 41] = -1   # Supervision on useless concepts (everything but red pills)
        if args.sup_version == '12sup':
            concepts[:, :30] = -1
        elif args.sup_version == '30sup':
            concepts[:, 30:] = -1

        out_dict = model(images)
        out_dict.update({'INPUTS': images, 'LABELS': labels, 'CONCEPTS': concepts, 'CE_WEIGHT': CE_weight, 'CE_weight_labels': CE_weight_labels})
        if use_prior and args.model in VAE_MODELS:
                mask = (concepts[0,:] != -1)
                prior = model.compute_prior(labels)
                out_dict.update({'mu_cluster': prior})
                predicted_mus = torch.tanh(out_dict['mu_cluster']/2)
                loss_bin = nn.BCELoss(reduction='mean')((1+predicted_mus[:, mask])/2, concepts.to(dtype = torch.float)[:, mask]) * prior_weight
                eval_losses['prior_loss'] = loss_bin.item()

        if args.sup_version == '12sup' and args.model in VAE_MODELS:
            out_dict['MUS'], out_dict['LOGVARS'] = out_dict['MUS'][:, :30], out_dict['LOGVARS'][:, :30]
            if use_prior and args.model in VAE_MODELS:
                out_dict['mu_cluster'] = out_dict['mu_cluster'][:, :30]
        elif args.sup_version == '30sup' and args.model in VAE_MODELS:
            out_dict['MUS'], out_dict['LOGVARS'] = out_dict['MUS'][:, 30:], out_dict['LOGVARS'][:, 30:]
            if use_prior and args.model in VAE_MODELS:
                out_dict['mu_cluster'] = out_dict['mu_cluster'][:, 30:]



        out_dict['std']= 0.0    # No noise in validation
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
    
    return eval_losses, cacc, acc, dci_ingredients           # inverted order of cacc and acc


