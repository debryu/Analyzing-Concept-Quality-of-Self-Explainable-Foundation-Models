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
from utils.utils import Leakage_Dataset, CS_estimator_Dataset
from sklearn.metrics import classification_report, accuracy_score
import math
from scipy.stats import entropy
from metrics.dci_framework import _compute_dci
from train_CS_classifiers import leakage_completeness_score
import pickle

#MODELS_TO_RUN = ['betaglancenet','cbmbase']
SKIP_MODELS = []
VERSIONS_TO_RUN = ['fullsup','12sup','30sup','nolabel']
SKIP_BETAS = []
SKIP_SEEDS = ['seedNone']

''' 
    Run different model to collect:

    1) frequency of labels classes to compute the entropy H(Y)
    2) the values of the input after the encoding layer to then train
    a classifier to predict the label and so estimate the entropy H(Y|C)
    3) the concepts ground truth
    4) the concepts predictions
       3) and 4) are used to compute to train a model to predict the label from the concepts and estimate the entropy H(Y|C)
    5) the concepts CE loss w.r.t. concept supervision to obtain the upper bound H(C|E)
'''

def create_folder(dataset, model, version, beta, seed, path = 'data/ckpt/saved_models/evaluated'):
  '''
    Create the folder to store the data
  '''
  
  folder = path + f'/{dataset}_{model}_{version}_{beta}_{seed}'
  if not os.path.exists(folder):
    os.makedirs(folder)
  return folder

def update_dictionary(out_dict, dictionary_to_update, losses):
  '''
    Create the dataset to estimate the COMPLETENESS SCORE
    The keys are:
    - all_labels
    - all_labels_predictions
    - all_concepts
    - all_concepts_predictions
    - all_images
    - all_encoder
    - concept_loss
    - label_loss
  '''
  dictionary_to_update['all_labels'].append(out_dict['LABELS'].detach().to('cpu').numpy())
  dictionary_to_update['all_labels_predictions'].append(out_dict['PREDS'].detach().to('cpu').numpy())
  dictionary_to_update['all_concepts'].append(out_dict['CONCEPTS'].detach().to('cpu').numpy())
  enc = out_dict['ENCODER'].detach().to('cpu')
  dictionary_to_update['all_encoder'].append(enc.reshape(enc.shape[0],-1).numpy())
  dictionary_to_update['all_images'].append(out_dict['INPUTS'].detach().to('cpu').numpy())
  dictionary_to_update['all_concepts_predictions'].append(torch.sigmoid(out_dict['LATENTS']).detach().to('cpu').numpy())
  # Losses
  dictionary_to_update['concept_loss'].append(losses['c-loss'])
  dictionary_to_update['label_loss'].append(losses['pred-loss'])
  
  return dictionary_to_update

def process_dictionary(dictionary):
  '''
    Process the dictionary to obtain the data in the right format
  '''

  keys = dictionary.keys()
  for key in keys:
    print(key)
    # Keep the list of losses, no need to average yet
    if key not in ['concept_loss','label_loss','dci_ingredients']:
      # Concatenate the list of numpy arrays into a single numpy array
      dictionary[key] = np.concatenate(dictionary[key])

  return dictionary

def run_DCI(dict, train_test_ratio = 0.7, fast = True):
  '''
    Run the DCI disentanglement metric
  '''
  (c_pred, g_true) = dict['dci_ingredients']
  L = len(g_true)
  #Randomly shuffle the data to prevent having only one class in the training set
  idx = np.random.permutation(L)
  c_pred = c_pred[idx]
  g_true = g_true[idx]

  z_train = c_pred[: round(L*train_test_ratio)]
  z_test  = c_pred[round(L*train_test_ratio) :]
  g_train = g_true[: round(L*train_test_ratio)]
  g_test  = g_true[round(L*train_test_ratio) :]
  # To make it faster
  if fast:
    z_train = c_pred[: 1000]
    z_test  = c_pred[1000 :2000]
    g_train = g_true[: 1000]
    g_test  = g_true[1000 :2000]

  dci_metrics = _compute_dci(z_train.T, g_train.T, z_test.T, g_test.T)
  return dci_metrics

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

def evaluate(model, loader, loss, model_type, ver, beta, seed, args, loc='test'):
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
        out_dict.update({'INPUTS': images, 'LABELS': labels, 'CONCEPTS': concepts})
        batch_loss, losses = loss(out_dict, args)

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

    return eval_losses, cacc, acc, process_dictionary(keep_track_of)

def find_models(models,dl,args):
  '''
    Find and evaluate the model
  '''
  output = {}
  out_dict = {}
  
  checkpoints = get_saved_models_to_run()
  for version in checkpoints:
    output[version] = {}
    model_type = version.split('_')[0]    # Example betaglancenet
    if model_type == 'betavae':
      # Since when I use betavae is just betaglancet without concept supervision, I can use the same model
      model = models['betaglancenet']
    else:
      model = models[model_type]
    ver = version.split('_')[1]           # Example partialconceptsupervision
    # Only run some versions
    if ver not in VERSIONS_TO_RUN:
      print(f'Skipping {model_type}-{ver}.')
      continue
    if model_type in SKIP_MODELS:
      print(f'Skipping {model_type}-{ver}.')
      continue
    for ckpt in checkpoints[version]:
      beta = ckpt.split('_')[3]             # Example beta1.0.pt
      beta = beta.split('.pt')[0]
      seed = ckpt.split('_')[2]             # Example seed90

      # Only run some seeds
      if seed not in SKIP_SEEDS:
       if beta in SKIP_BETAS:
        continue
       else:
        print(f'Running {version}-{ckpt}')
        path = checkpoints[version][ckpt]
        loss  = model.get_loss(args)
        print(path)
        model.load_state_dict(torch.load(path))
        model.to(model.device)
        print('--->    Chosen device: ', model.device, '\n')

        output[version][ckpt] = {}
        _, concept_acc, label_acc, stats_dict = evaluate(model, dl, loss, model_type, ver, beta, seed, args, args.split)
        if model_type not in out_dict:
                out_dict[model_type] = {}
        if version not in out_dict[model_type]:
            out_dict[model_type][ver] = {}
        if beta not in out_dict[model_type][ver]:
            out_dict[model_type][ver][beta] = {}
        if seed not in out_dict[model_type][ver][beta]:
            out_dict[model_type][ver][beta][seed] = {}

        concepts = torch.tensor(stats_dict['all_concepts_predictions'])
        labels = torch.tensor(stats_dict['all_labels'])
        encoders = torch.tensor(stats_dict['all_encoder'])

        train_proportion = 0.8
        dataset_size_train = math.floor(len(labels)*train_proportion)
        train_conc = concepts[:dataset_size_train,:]
        train_labels = labels[:dataset_size_train]
        train_encoders = encoders[:dataset_size_train,:]
        test_conc = concepts[dataset_size_train:,:]
        test_labels = labels[dataset_size_train:]
        test_encoders = encoders[dataset_size_train:,:]

        # Compute dataset frequencies for class 0 and 1
        classes_train = train_labels.unique()
        classes_test = test_labels.unique()
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
        output[version][ckpt]['concept_loss'] = f'Concept CE Loss train: {CE_concepts_train} - Concept CE Loss test: {CE_concepts_test}'
        # Create the CS estimation dataset
        train_ds = CS_estimator_Dataset(train_conc, train_labels, train_encoders, {'H_Y': H_Y_train,
                                                                  'CE_concepts': CE_concepts_train,
                                                                  })
        test_ds = CS_estimator_Dataset(test_conc, test_labels, test_encoders, {'H_Y': H_Y_test,
                                                                'CE_concepts': CE_concepts_test,
                                                                })
        CS_dataset = { 'train': train_ds, 'test': test_ds}

        # In order to create the leakage estimation dataset, remove the two important concepts for prediction (red, pill)
        indexes_to_remove = [20,21,22,23,24,25,26,27,28,29,38,39,40,41]  # Create a tensor with all 1 except for the masked concepts
        #indexes_to_remove = []
        relevant_concepts_mask = torch.ones(train_conc.shape[1], dtype=torch.bool)
        relevant_concepts_mask[indexes_to_remove] = False
        # Create the Leakage estimation dataset
        train_ds = Leakage_Dataset(train_conc[:,relevant_concepts_mask], train_labels) # Remove the masked concepts on dimension 1
        test_ds = Leakage_Dataset(test_conc[:,relevant_concepts_mask], test_labels)   # Remove the masked concepts on dimension 1
        LKG_dataset = { 'train': train_ds, 'test': test_ds}

        # Save the datasets
        torch.save(CS_dataset, os.path.dirname(os.path.realpath(__file__)) + f'/data/encoder_dataset/CS_{ver}_{model_type}_{seed}_{beta}.pth')
        torch.save(LKG_dataset, os.path.dirname(os.path.realpath(__file__)) + f'/data/encoder_dataset/LKG_{ver}_{model_type}_{seed}_{beta}.pth')
        experiment_folder = create_folder(args.dataset, model_type, ver, beta, seed)
        # Make sure there is no overlapping
        if os.path.exists(experiment_folder + '/CS_dataset.pth') or os.path.exists(experiment_folder + '/LKG_dataset.pth'):
          print('Trying to create the file: ', experiment_folder + '/CS_dataset.pth and .../LKG_dataset.pth')
          raise ValueError('The file already exists!')
        torch.save(CS_dataset, experiment_folder + '/CS_dataset.pth')
        torch.save(LKG_dataset, experiment_folder + '/LKG_dataset.pth')

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
        
        # Compute the completeness metric
        cs_havasi_mi = completeness_havasi_mi(stats_dict['all_labels'], stats_dict['all_concepts'], stats_dict['all_images'])
        print('Completeness score Havasi-MI: ', cs_havasi_mi)
        
        # Run DCI
        DCI = run_DCI(stats_dict, fast = True)
        DCI_disentanglement = DCI['disentanglement']
        



        # Print the desired metrics
        # CE concepts
        output[version][ckpt]['CE_concepts'] = np.mean(stats_dict['concept_loss'])
        # CE labels
        output[version][ckpt]['CE_labels'] = np.mean(stats_dict['label_loss'])
        # classification reports
        output[version][ckpt]['concept_report'] = concepts_report
        output[version][ckpt]['label_report'] = labels_report
        # Only the macro avg precision
        output[version][ckpt]['concepts_macroavgprecision'] = concepts_report['macro avg']['precision']
        output[version][ckpt]['labels_macroavgprecision'] = labels_report['macro avg']['precision']
        # Entropy H(Y) (used for Completeness Score)
        output[version][ckpt]['entropy'] = f'Entropy H(Y) train: {H_Y_train} - Entropy H(Y) test: {H_Y_test}'
        # DCI
        output[version][ckpt]['DCI'] = DCI
        # DCI disentanglement
        output[version][ckpt]['DCI_disentanglement'] = DCI_disentanglement

        some_metrics = {'CE_concepts': np.mean(stats_dict['concept_loss']),
                        'CE_labels': np.mean(stats_dict['label_loss']),
                        'DCI_disentanglement': DCI_disentanglement,
                        'DCI': DCI,
                        'concept_report': concepts_report,
                        'label_report': labels_report,
                        }
        out_dict[model_type][ver][beta][seed].update(some_metrics)

        # Create the folder to store the results
        experiment_folder = create_folder(args.dataset, model_type, ver, beta, seed)
        # Make sure there is no overlapping
        if os.path.exists(experiment_folder + '/metrics_dict_part1.pkl'):
          print('Trying to create the file: ', experiment_folder + '/metrics_dict_part1.pkl')
          raise ValueError('The file already exists!')
        pickle.dump(some_metrics, open(experiment_folder + '/metrics_dict_part1.pkl', 'wb'))

        ## REMOVE MAYBE?
        output[version][ckpt]['concept_accuracy'] = concept_acc
        output[version][ckpt]['label_accuracy'] = label_acc
        
        #output[version][ckpt]['concept_report'] = concepts_report
        #output[version][ckpt]['label_report'] = labels_report
        #output[version][ckpt]['completeness_score'] = cs_havasi_mi
        #output[version][ckpt]['concepts_macroavgprecision'] = concepts_report['macro avg']['precision']
        #output[version][ckpt]['labels_macroavgprecision'] = labels_report['macro avg']['precision']
      else:
        print(f'Skipping {seed} for {model_type}-{ver}-{beta}.')
  return output, out_dict
      
        

  
parser = argparse.ArgumentParser(description='Run and Evaluate models', allow_abbrev=False)

torch.set_num_threads(4)
parser = add_args(parser)
# Get the arguments
args = parser.parse_args()
args.num_C, args.num_Y = get_dataset_C_Y(args)
# Import the dataset
dataset = get_dataset(args)

# PARAMETERS OF THE MODEL
args.dataset = 'shapes3d'
args.latent_dim = 42

used_models = ['betaglancenet','cbmbase']
models = {}
for m in used_models:
  args.model = m
  mod = importlib.import_module('models.' + args.model)
  encoder, decoder  = dataset.get_backbone(args)
  models[m] = get_model(args, encoder, decoder)

# Import the dataloaders
train_loader, val_loader, (test_loader, ood_loader) = dataset.get_data_loaders()
if args.split == 'val':
  data_loader = val_loader
elif args.split == 'ood':
  data_loader = ood_loader
else:
  data_loader = test_loader

res, out_dict = find_models(models,data_loader,args)
versions = res.keys()

leakage_and_CS = leakage_completeness_score(n_epochs=100, skip_models=SKIP_MODELS,skip_versions=['badsup'],skip_betas=SKIP_BETAS,skip_seeds=SKIP_SEEDS, args=args)




for version in versions:
   checkpoints = res[version].keys()
   for check in checkpoints:
      print('________________________________________________________________________________________________________________________________________________________________________')
      print(f'-------------------------------- [ Version: {version} - Checkpoint: {check} ] ----------------------------')
      print('________________________________________________________________________________________________________________________________________________________________________')
      print(f'Entropy: {res[version][check]["entropy"]}')
      print(f'DCI Disentanglement: {res[version][check]["DCI_disentanglement"]}')
      print(f'DCI: {res[version][check]["DCI"]}')
      print(f'Split CE Loss: {res[version][check]["concept_loss"]}')
      print(f'CE_concepts: {res[version][check]["CE_concepts"]}')
      print(f'CE_labels: {res[version][check]["CE_labels"]}')
      print(f'Concept Accuracy: {res[version][check]["concept_accuracy"]}')
      print(f'Concepts Macro Avg Precision: {res[version][check]["concepts_macroavgprecision"]}')
      print(f'Label Accuracy: {res[version][check]["label_accuracy"]}')
      print(f'Labels Macro Avg Precision: {res[version][check]["labels_macroavgprecision"]}')
      print('\n')
      print(f'Concepts Classification Report: {res[version][check]["concept_report"]}')
      print('\n')
      print(f'Labels Classification Report: {res[version][check]["label_report"]}')
      print('\n')
      print('______________________________________________________________________________________________________________________________________________________________________')
      print('\n')
      print('\n')


print('\n \n \n')
latex_table = ''
for model_type in out_dict.keys():
  for version in out_dict[model_type].keys():
     for beta in out_dict[model_type][version].keys():
        for seed in out_dict[model_type][version][beta].keys():
          DCI_dis_latex = out_dict[model_type][version][beta][seed]['DCI_disentanglement']
          CS_latex = leakage_and_CS[model_type][version][beta][seed]['CS']
          Leakage_CE_latex = leakage_and_CS[model_type][version][beta][seed]['CE_labels']
          Leakage_f1_latex = leakage_and_CS[model_type][version][beta][seed]['f1-score-leakage']
          CE_concepts_latex = out_dict[model_type][version][beta][seed]['CE_concepts']
          CE_labels_latex = out_dict[model_type][version][beta][seed]['CE_labels']
          f1_score_labels_latex = out_dict[model_type][version][beta][seed]['label_report']['macro avg']['f1-score']
          latex_table += f'{model_type} $\\{beta}$ & {version} & {DCI_dis_latex:.3f} & {CS_latex:.3f} & {Leakage_CE_latex:.1e} & {Leakage_f1_latex:.3f} & {CE_concepts_latex:.1e} & {f1_score_labels_latex:.3f} \\\\ \n'
          latex_table += '\\hline \n'
print(latex_table)
print('\n \n \n')