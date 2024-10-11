import os
import pickle
import argparse
import importlib
from datasets import get_dataset
from models import get_model
from utils.utils import get_dataset_C_Y
import torch
from utils.preprocessing_model_data import update_dictionary, process_dictionary, INDEXES_TO_REMOVE, update_indices, leakage_indices, dci_indices
import math
from scipy.stats import entropy
import numpy as np
from utils.estimators.estimator_model import Leakage_Dataset, CS_estimator_Dataset, leakage, completeness_score
from sklearn.metrics import classification_report
from metrics.dci_framework import _compute_dci

ALLOWED_VERSIONS = ['fullsup','nolabel','12sup','30sup','badsup']


def run_DCI(dict, train_test_ratio = 0.7, mode = 'fast'):
  '''
    Run the DCI disentanglement metric
  '''
  (c_pred, g_true) = dict['dci_ingredients']
  assert g_true.shape[0] == c_pred.shape[0] # Make sure the number of samples is the same
  L = g_true.shape[0]
  #Randomly shuffle the data to prevent having only one class in the training set
  idx = np.random.permutation(L)
  #print(idx)
  c_pred = c_pred[idx]
  g_true = g_true[idx]

  z_train = c_pred[: round(L*train_test_ratio)]
  z_test  = c_pred[round(L*train_test_ratio) :]
  g_train = g_true[: round(L*train_test_ratio)]
  g_test  = g_true[round(L*train_test_ratio) :]
  # To make it faster
  if mode == 'fast':
    print('Running DCI in fast mode (training on only 2000 samples)')
    z_train = c_pred[: 1000]
    z_test  = c_pred[2000 :3000]
    g_train = g_true[: 1000]
    g_test  = g_true[2000 :3000]
  
  if mode == 'debug':
    print('Running DCI in debug mode (training on only 5 samples)')
    z_train = c_pred[: 5]
    z_test  = c_pred[5 :8]
    g_train = g_true[: 5]
    g_test  = g_true[5 :8]

  dci_metrics = _compute_dci(z_train.T, g_train.T, z_test.T, g_test.T)
  return dci_metrics

def create_CS_LKG_dataset(stats_dict:dict, save_path, args, train_proportion:float = 0.7):
    # Get the values from the model
    concepts = torch.tensor(stats_dict['all_concepts_predictions'])
    labels = torch.tensor(stats_dict['all_labels'])
    encoders = torch.tensor(stats_dict['all_encoder'])
    
    # Split the dataset into train and test
    dataset_size_train = math.floor(len(labels)*train_proportion)
    train_conc = concepts[:dataset_size_train,:]
    train_labels = labels[:dataset_size_train]
    train_encoders = encoders[:dataset_size_train,:]
    test_conc = concepts[dataset_size_train:,:]
    test_labels = labels[dataset_size_train:]
    test_encoders = encoders[dataset_size_train:,:]

    # Compute dataset frequencies for class 0 and 1
    classes_train = train_labels.unique()
    print(classes_train)
    classes_test = test_labels.unique()
    print(classes_test)
    frequency_train = (train_labels == classes_train[1]).sum().item() / len(train_labels)
    frequency_test = (test_labels == classes_test[1]).sum().item() / len(test_labels)
    print(f'Frequency of class 1 in train: {frequency_train} - Frequency of class 1 in test: {frequency_test}')
    
    # Compute H(Y) 
    H_Y_train = entropy([frequency_train, 1-frequency_train], base=2)
    H_Y_test = entropy([frequency_test, 1-frequency_test], base=2)
    print(f'Entropy H(Y) train: {H_Y_train} - Entropy H(Y) test: {H_Y_test}')

    # Compute the CE loss for the concepts w.r.t. the concept supervision
    len_ce_concepts = len(stats_dict['concept_loss'])
    ce_concepts_ds_size = math.floor(len_ce_concepts * train_proportion)
    CE_concepts_train = np.mean(stats_dict['concept_loss'][:ce_concepts_ds_size])
    CE_concepts_test = np.mean(stats_dict['concept_loss'][ce_concepts_ds_size:])
    print(f'Concept CE Loss train: {CE_concepts_train} - Concept CE Loss test: {CE_concepts_test}')

    # Create the CS estimation dataset
    train_ds = CS_estimator_Dataset(train_conc, train_labels, train_encoders, {'H_Y': H_Y_train,
                                                              'CE_concepts': CE_concepts_train,
                                                              'frequency': frequency_train,
                                                              })
    test_ds = CS_estimator_Dataset(test_conc, test_labels, test_encoders, {'H_Y': H_Y_test,
                                                            'CE_concepts': CE_concepts_test,
                                                            'frequency': frequency_test,
                                                            })
    CS_dataset = { 'train': train_ds, 'test': test_ds}
    torch.save(CS_dataset,os.path.join(save_path,'CS_dataset.ds'))

    # Create the Leakage estimation dataset

    # Remove concepts to train the leakage model
    
    if args.dataset not in INDEXES_TO_REMOVE.keys():
       raise NotImplementedError(f'Not defined any indexes to remove for the dataset {args.dataset}.')
    else:
        indexes_to_remove = INDEXES_TO_REMOVE[args.dataset] # This is the list of concepts to remove
        # w.r.t. the original concept list
        # We need to update the indexes to match the concept list used by the model (after clip_cutoff and interpretability_cutoff)
        if os.path.exists(os.path.join(save_path,'removed_concepts_id.list')):
            removed_already = pickle.load(open(os.path.join(save_path,'removed_concepts_id.list'),'rb'))
            print('Creating LKG ds -> Concepts to remove due to lf-cmb filtering:', len(removed_already), removed_already)
        else:
            print('Could not find', os.path.join(save_path,'removed_concepts_id.list'), '. Empty list created')
            removed_already = []
        indexes_to_remove = update_indices(indexes_to_remove, removed_already)
    
    print('Removed concepts:', indexes_to_remove, len(indexes_to_remove))
    print('removed_already:', removed_already, len(removed_already))

    #print(indexes_to_remove)
    print('Removed concepts:', indexes_to_remove, len(indexes_to_remove))
    relevant_concepts_mask = torch.ones(train_conc.shape[1], dtype=torch.bool)
    relevant_concepts_mask[indexes_to_remove] = False

    print(train_conc[:,relevant_concepts_mask].shape)
    print(train_labels.shape)

    id_to_lkg, lkg_to_id = leakage_indices(train_conc.shape[1], indexes_to_remove)
    count = 100
    if 'lfcbm_bottleneckSize' in stats_dict.keys():
        count = stats_dict['lfcbm_bottleneckSize']
    id_to_dci, dci_to_id = dci_indices(count, indexes_to_remove)
    
    explainable_lkg = {'id_to_lkg': id_to_lkg, 'lkg_to_id': lkg_to_id, 'id_to_dci': id_to_dci, 'dci_to_id': dci_to_id}
    pickle.dump(explainable_lkg, open(os.path.join(save_path,'explainable_indexes_lkg.dict'),'wb'))

    train_ds = Leakage_Dataset(train_conc[:,relevant_concepts_mask], train_labels) # Remove the masked concepts on dimension 1
    test_ds = Leakage_Dataset(test_conc[:,relevant_concepts_mask], test_labels)   # Remove the masked concepts on dimension 1
    LKG_dataset = { 'train': train_ds, 'test': test_ds}
    torch.save(LKG_dataset,os.path.join(save_path,'LKG_dataset.ds'))
    return 

def compute_metrics(stats_dict:dict, save_folder, fast = True):
    # Run the a full accuracy-precision-recall-f1 report
    concepts_pred_list = (stats_dict['all_concepts_predictions'] > 0.5).astype(int).tolist()
    concepts_gt_list = stats_dict['all_concepts'].astype(int).tolist()
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
    labels_pred_list = stats_dict['all_labels_predictions'].argmax(axis = 1).tolist()
    labels_gt_list = stats_dict['all_labels'].tolist()
    concepts_report = classification_report(concepts_gt_list, concepts_pred_list, output_dict=True)
    labels_report = classification_report(labels_gt_list, labels_pred_list, target_names = ['label_0', 'label_1'], output_dict=True)
    print('Concepts macro avg precision: ', concepts_report['macro avg']['precision'])
    print('Labels macro avg precision: ', labels_report['macro avg']['precision'])

    # Run DCI
    DCI = run_DCI(stats_dict, fast)
    DCI_disentanglement = DCI['disentanglement']
    metrics_dict = {'DCI':DCI, 
                    'DCI_disentanglement': DCI_disentanglement,
                    'CE_concepts': np.mean(stats_dict['concept_loss']),
                    'CE_labels': np.mean(stats_dict['label_loss']),
                    'concept_report': concepts_report,
                    'label_report': labels_report,
                    }
    
    pickle.dump(metrics_dict, open(os.path.join(save_folder,'metrics.dict'), 'wb'))
    return metrics_dict

def evaluate(model, loader, loss, ver, args, save_folder, loc='test', ds_freq = None): #Remove loc
    '''
    Evaluate the model on the test loader (or val)
    to extract the losses, predictions, encoder outputs, concepts and metrics
    '''
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
    all_encoder = []
    concept_loss = []

    keep_track_of = {}
    keep_track_of['all_labels'] = []
    keep_track_of['all_labels_predictions'] = []
    keep_track_of['all_concepts'] = []
    keep_track_of['all_concepts_predictions'] = []
    keep_track_of['all_images'] = []
    keep_track_of['all_encoder'] = []
    keep_track_of['concept_loss'] = []
    keep_track_of['label_loss'] = []

    for i, data in enumerate(loader):
        images, labels, concepts = data 
        images, labels, concepts = images.to(model.device), labels.to(model.device), concepts.to(model.device)
        original_concepts = concepts.clone()
        ''' CONTROL CONCEPTS SUPERVISION HERE '''
        # Mask the concepts
        # Only keep scale and shape, not color (orientation already masked in the dataset)
        #{'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 'scale': 8, 'shape': 4, 'orientation': 15}
        
        if ver == '12sup':
          concepts[:, :30] = -1
        if ver == '30sup':
          concepts[:, 30:] = -1
        if ver == 'badsup':
          concepts[:, 20] = -1
          concepts[:, 41] = -1
        
        # Run the model
        out_dict = model(images)
        #print(out_dict)
        #print(labels)
        CE_weight = ds_freq['CE_weight']
        frequencies = ds_freq['frequencies']
        CE_weight_labels = ds_freq['CE_weight_labels']

        out_dict.update({'INPUTS': images, 'LABELS': labels, 'CONCEPTS': concepts, 'CE_WEIGHT': CE_weight, 'CE_weight_labels': CE_weight_labels})
        #out_dict.update({'INPUTS': images, 'LABELS': labels, 'CONCEPTS': concepts})
        batch_loss, losses = loss(out_dict, args)
        
        # Restore concepts
        out_dict.update({'CONCEPTS': original_concepts})

        #print(out_dict['PREDS'])
        #print(out_dict['LABELS'])
        # Collect the labels, concepts and images for the completeness metric
        keep_track_of = update_dictionary(out_dict, keep_track_of, losses)
        #all_labels.append(labels.detach().to('cpu').numpy())
        #all_concepts.append(concepts.detach().to('cpu').numpy())
        #enc = out_dict['ENCODER'].detach().to('cpu')
        #all_encoder.append(enc.reshape(enc.shape[0],-1).numpy())
        #all_images.append(images.detach().to('cpu').numpy())
        
        #batch_loss, losses = loss(out_dict, args)
        #concept_loss.append(losses['c-loss'])
        
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
        
        '''Don't remove this, may come useful'''
        # assign 0 to the concept if it is less than 0.5, 1 otherwise
        #c_predictions = (c_predictions > 0.5).type(torch.float)
        '''Don't remove this, may come useful'''

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
            raise NotImplementedError('Not implemented yet')
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
    keep_track_of['dci_ingredients'] = dci_ingredients

    stats_dict = process_dictionary(keep_track_of)
    # Do not save the internal state dict, it is too big
    #pickle.dump(stats_dict, open(os.path.join(save_folder,'internal_state.dict'), 'wb'))
    return stats_dict


def run_model(model, model_metadata:dict, data_loader, args:argparse.Namespace):
  '''
    Run the model on the data_loader
  '''
  loss  = model.get_loss(args)
  
  # Check if LKG and CS datasets are needed
  if os.path.exists(model_metadata['folder'] + '/CS_dataset.ds') and os.path.exists(model_metadata['folder'] + '/LKG_dataset.ds') and os.path.exists(model_metadata['folder'] + '/metrics.dict'):
      print('LKG and CS datasets already exist! - Loading previous results')
      #stats_dict = pickle.load(open(os.path.join(model_metadata['folder'],'internal_state.dict'),'rb'))    # This is not saved, as it is too big
      metrics = pickle.load(open(model_metadata['folder'] + '/metrics.dict', 'rb'))
  else:
      stats_dict = evaluate(model, data_loader, loss, model_metadata['version'], args, model_metadata['folder'], 'test', model_metadata['ds_freq'])
      create_CS_LKG_dataset(stats_dict, model_metadata['folder'], args)
      metrics = compute_metrics(stats_dict, model_metadata['folder'], fast = True)

  '''
  CHANGE FAST to False TO HAVE IT COMPUTE DCI MORE PRECISELY
  '''
  if os.path.exists(model_metadata['folder'] + '/LKG_results.dict') and os.path.exists(model_metadata['folder'] + '/CS_results.dict'):
      print('LKG and CS results already exist! - Loading previous results')
      leakage_results = pickle.load(open(model_metadata['folder'] + '/LKG_results.dict', 'rb'))
      CS_results = pickle.load(open(model_metadata['folder'] + '/CS_results.dict', 'rb'))
  else:
      leakage_results = leakage(model_metadata,args)
      CS_results = completeness_score(model_metadata)

  metrics.update(leakage_results)
  metrics.update(CS_results)
  return metrics

def run_analysis(model_metadata:dict, exclude:dict = {}, split = 'test'):
  '''
    Run the analysis for a specific model (by specifying the folder path)
  '''
  # Collect the args
  args = pickle.load(open(model_metadata['args'], 'rb'))
  # Check if the model should be excluded
  for key in exclude:
    if key in vars(args).keys():  #Convert the args to a dictionary to get the keys
      if getattr(args,key) == exclude[key]:
        print(f'Skipping model {args.model} because of {key} = {exclude[key]}')
        return
    if key in model_metadata.keys():
      if model_metadata[key] == exclude[key]:
        print(f'Skipping model {args.model} because of {key} = {exclude[key]}')
        return

  # Extract the model type
  model_type = args.model
  # Get the dataset
  dataset = get_dataset(args)
  model_metadata.update({'dataset': args.dataset}) # Add the dataset to the metadata
  ds_string = args.dataset
  # Instatiate the model
  mod = importlib.import_module('models.' + args.model)
  encoder, decoder  = dataset.get_backbone(args)
  model = get_model(args, encoder, decoder)

  model.load_state_dict(torch.load(model_metadata['checkpoint']))
  model.to(model.device)
  #print('--->    Chosen device: ', model.device, '\n')

  # Import the dataloaders
  train_loader, val_loader, (test_loader, ood_loader) = dataset.get_data_loaders()
  print(f'Running analysis for model {model_type}:')
  print(args)
  if split == 'val':
    data_loader = val_loader
  elif split == 'ood':
    data_loader = ood_loader
  else:
    data_loader = test_loader
  
  # Load the dataset frequencies
  CE_weight = None
  CE_weight_labels = None
  if os.path.exists(f'data/ds_freqs/{ds_string}_freq.pkl'):
      ds_freq = pickle.load(open(f'data/ds_freqs/{ds_string}_freq.pkl', 'rb'))
      CE_weight = ds_freq['CE_weight']
      frequencies = ds_freq['frequencies']
      CE_weight_labels = ds_freq['CE_weight_labels']
      print('Concept CE weight:', CE_weight)
      print('Concept freq:', frequencies)
      print('CE_weight_labels:', CE_weight_labels)
  else:
      frequencies = torch.zeros(args.latent_dim).to(model.device)
      freq_labels = torch.zeros(2).to(model.device)
      n_sums = 0
      # Compute concept frequencies
      for i, data in enumerate(tqdm(train_loader)):
          images, labels, concepts = data #Should be fixed now
          images, labels, concepts = images.to(model.device), labels.to(model.device), concepts.to(model.device)
          n_concepts = concepts.shape[1]
          batch_size = concepts.shape[0]
          frequencies += torch.sum(concepts, dim=0).float()
          freq_labels[1] += torch.sum(labels).float()
          n_sums += batch_size
      
      # If frequency is 0, set it to 100000000 to avoid division by 0
      frequencies[frequencies == 0] = 100000000

      CE_weight = n_sums/frequencies
      freq_labels[0] = n_sums - freq_labels[1]
      CE_weight_labels = n_sums/freq_labels
      frequencies = frequencies/n_sums
      ds_freq = {'CE_weight': CE_weight, 'frequencies': frequencies, 'CE_weight_labels': CE_weight_labels, 'freq_lables': freq_labels}
      pickle.dump(ds_freq, open(f'data/ds_freqs/{ds_string}_freq.pkl', 'wb'))

      print('Concept CE weight:', CE_weight)
      print('Concept freq:', frequencies)
      print('CE_weight_labels:', CE_weight_labels)
  
  model_metadata['ds_freq'] = ds_freq

  # Run the model
  metrics = run_model(model, model_metadata, data_loader, args)
  # Add also the model metadata to the metrics
  metrics.update(model_metadata)
  # Merge also the args
  metrics.update(vars(args))
  # Save the metrics
  pickle.dump(metrics, open(os.path.join(model_metadata['folder'],'complete_analysis.dict'), 'wb'))
  
  return metrics

def get_saved_models_to_run(saved_models_folder = '/data/ckpt'):
  '''
    Get all the different model checkpoints (paths) to run from the models folder
  '''
  # Folder where all the checkpoints are stored
  checkpoints_path = saved_models_folder
  # Get a list of all folders inside the folder
  variations =  [f for f in os.listdir(checkpoints_path) if not f.endswith ('.pt') and not f.endswith('models')]
  
  models = {}
  # Iterate over all the folders containing the different models
  for v in variations:
      version = None
      args = None
      checkpoint = None
      #print(saved_models_folder)
      #print(v)
      checkpoint_path = os.listdir(os.path.join(checkpoints_path,v))
      
      # Iterate over all the files in the folder
      for f in checkpoint_path:
          if f.endswith('txt'):
            version = f.split('.')[0]
          if f.endswith('pt'):
            checkpoint = os.path.join(checkpoints_path,v,f)
          if f.endswith('pkl'):
            args = os.path.join(checkpoints_path,v,f)
      
      # Check if version is one of the allowed
      if version not in ALLOWED_VERSIONS:
          print(v)
          print(f'Error: Model version {version} not allowed')
          raise Exception(f'Error: Model version {version} not allowed')
      # If all the necessary files are present, add the model to the list
      if version!=None and args!=None and checkpoint!=None:
        models[v] = {'version': version, 'args': args, 'checkpoint': checkpoint, 'folder': os.path.join(checkpoints_path,v)}
      else:
        print(f'Error: Missing files for model {v}')
        raise Exception(f'Error: Missing files for model {v}')
      
  return models


