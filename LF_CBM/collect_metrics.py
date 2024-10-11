import os
import torch
import pickle 
from tqdm import tqdm 
from utils.data_utils import get_data
import argparse
import numpy as np
from sklearn.metrics import classification_report
from collect_metrics import create_CS_LKG_dataset, compute_metrics, run_DCI
from models.lfcbm import LFcbm, load_cbm
from utils.estimators.estimator_model import Leakage_Dataset, CS_estimator_Dataset, leakage, completeness_score, SUPERVISED_BOTTLENECK_DIM
#from utils.save_importance_matrix import save_IM_as_img
'''
TODO:
- [x] add checks to prevent recomputing metrics when they are already saved
- [x] add leakage and cs functions
- [] little cleanup
- [] in estimator_model.py, SUPERVISED_BOTTLENECK_DIM should be generated depending on the concepts generated

'''
shapes3d_concepts_set_dim = 87         

def evaluate_labo(all_concepts,all_labels, mask, dataset, loss, device, model_metadata, args, num_classes = 2):
    print("Evaluating LaBo")
    all_concepts_predictions = torch.load(os.path.join(model_metadata['folder'], "LaBo/val_concepts.labo"))
    print(all_concepts_predictions.shape)
    weights = torch.load(os.path.join(model_metadata['folder'] ,"LaBo/labo_weights.pt"), map_location=device)
    model = torch.nn.Linear(in_features=weights.shape[1], out_features=weights.shape[0], bias=False).to(device)
    
    model.load_state_dict({"weight":weights})
    model.eval()
    all_labels = all_labels.to(device)
    
    all_concepts_predictions = all_concepts_predictions.to(device)
    all_concepts_predictions = all_concepts_predictions[:,mask]
    all_labels_predictions = model(all_concepts_predictions)
    test_loss = loss(all_labels_predictions, all_labels)    
    avg_loss = test_loss.item()
    
    
    #print(all_concepts_predictions.shape)
    #print(all_labels.shape)
    #print(activation_encoder.shape)
    
    # Set this to zero, as we don't have supervision on the concepts
    concept_loss = np.zeros(all_concepts_predictions.shape[0])

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=f'{dataset}_lfcbm')
    args = parser.parse_args()
    # Run the a full accuracy-precision-recall-f1 report
    labels_pred_list = all_labels_predictions.argmax(axis = 1).tolist()
    labels_gt_list = all_labels.tolist()
    
    concepts_pred_list = all_concepts_predictions.squeeze()
    
    concepts_gt_list = all_concepts.squeeze()
    #print('pred_conc', concepts_pred_list.shape)
    num_C = concepts_pred_list.shape[1]
    #print('gt_conc', concepts_gt_list.shape)
    labels_report = classification_report(labels_gt_list, labels_pred_list, target_names = ['label_0', 'label_1'], output_dict=True)
    print('Labels f1-score: ', labels_report['macro avg']['f1-score'])

    concepts_gt_list = concepts_gt_list.cpu()
    concepts_pred_list = concepts_pred_list.cpu()
    print('LaBo C:',concepts_pred_list.shape)
    # Run DCI
    DCI = run_DCI({'dci_ingredients': (concepts_pred_list,concepts_gt_list)}, mode = 'fast')
    #print(DCI)
    im = torch.tensor(DCI['importance_matrix'])
    im = torch.nn.functional.softmax(im, dim=0)
    #print(im)
    #print(torch.min(im,dim = 0))
    #print(torch.max(im,dim = 0))
    DCI_disentanglement = DCI['disentanglement']
    
    bottleneck_lfcbm_dim = None
    if model_metadata['dataset'] in ['shapes3d','shapes3d_lfcbm']:
        bottleneck_lfcbm_dim = shapes3d_concepts_set_dim
    elif model_metadata['dataset'] in ['celeba','celeba_lfcbm']:
        bottleneck_lfcbm_dim = 99

    metrics_dict = {'DCI':DCI, 
                    'DCI_disentanglement': DCI_disentanglement,
                    'CE_concepts': None,
                    'CE_labels': avg_loss,
                    'concept_report': None,
                    'label_report': labels_report,
                    'num_C': num_C,
                    'folder': os.path.join(model_metadata['folder'] ,"LaBo"),
                    'dataset': model_metadata['dataset'],
                    'model': 'LaBo',
                    'lfcbm_bottleneckSize': bottleneck_lfcbm_dim,
                    'version': 'LaBo'
                    }
    
    activation_encoder = torch.load(os.path.join(model_metadata['activation_dir'], f"{dataset}_val_clip_RN50.pt"))

    os.makedirs(model_metadata['folder']+'/LaBo', exist_ok=True)
    print(all_concepts_predictions.shape)
    
    print("Creating for LaBo CS-LKG dataset with original concept set size: ", bottleneck_lfcbm_dim)
    create_CS_LKG_dataset({'all_concepts_predictions':all_concepts_predictions, 
                       'all_labels':all_labels, 
                       'all_encoder':activation_encoder,
                       'concept_loss':concept_loss,
                       'lfcbm_bottleneckSize': bottleneck_lfcbm_dim,
                       }, model_metadata['folder']+'/LaBo', args)
    metrics_dict.update({'lfcbm_bottleneckSize': bottleneck_lfcbm_dim})
    #save_IM_as_img(model_metadata['folder']+'/LaBo', DCI['importance_matrix'])

    pickle.dump(metrics_dict, open(os.path.join(model_metadata['folder'], 'LaBo/labo_metrics.dict'), 'wb'))
    
    return metrics_dict

def evaluate_lf(model, loader, loss, device, model_metadata, args):
    print("Evaluating LF")
    print(model_metadata['mask'])
    all_concepts_predictions = []
    all_concepts = []
    all_labels = []
    all_labels_predictions = []
    losses = []
    model.eval()
    for i,batch in enumerate(tqdm(loader)):
        images, labels, concepts = batch
        batch_size = labels.shape[0]
        images = images.to(device)
        labels = labels.to(device).to(torch.long)
        preds, pred_concepts = model(images)
        test_loss = loss(preds, labels)
        losses.append(test_loss.item())
        all_concepts_predictions.append(pred_concepts.detach().to('cpu'))
        all_labels.append(labels.detach().to('cpu')) 
        all_labels_predictions.append(preds.detach().to('cpu'))
        all_concepts.append(concepts.detach().to('cpu'))
        
        #if i == 10:
        #    break

    all_concepts = torch.cat(all_concepts, dim=0)
    all_concepts_predictions = torch.cat(all_concepts_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_labels_predictions = torch.cat(all_labels_predictions, dim=0)


    '''
    ALSO EVALUATE LABO
    '''
    LaBo_metrics = evaluate_labo(all_concepts, all_labels, model_metadata['mask'], model_metadata['dataset'], loss, device, model_metadata, args)


    #print(all_concepts_predictions.shape)
    #print(all_labels.shape)
    #print(activation_encoder.shape)
    concept_loss = np.zeros(all_concepts_predictions.shape[0])

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=f'{model_metadata["dataset"]}_lfcbm')
    args = parser.parse_args()
    # Run the a full accuracy-precision-recall-f1 report
    labels_pred_list = all_labels_predictions.argmax(axis = 1).tolist()
    labels_gt_list = all_labels.tolist()
    concepts_pred_list = all_concepts_predictions.squeeze()
    concepts_gt_list = all_concepts.squeeze()
    print('GT concepts shape', concepts_gt_list.shape)
    #print('pred_conc', concepts_pred_list.shape)
    num_C = concepts_pred_list.shape[1]
    #print('gt_conc', concepts_gt_list.shape)
    labels_report = classification_report(labels_gt_list, labels_pred_list, target_names = ['label_0', 'label_1'], output_dict=True)
    print('Labels f1-score: ', labels_report['macro avg']['f1-score'])

    #print('AAAAD',concepts_pred_list.shape)
    #asd
    # Run DCI
    DCI = run_DCI({'dci_ingredients': (concepts_pred_list,concepts_gt_list)}, mode = 'fast')
    #print(DCI)
    im = torch.tensor(DCI['importance_matrix'])
    im = torch.nn.functional.softmax(im, dim=0)
    #print(im)
    #print(torch.min(im,dim = 0))
    #print(torch.max(im,dim = 0))
    #save_IM_as_img(model_metadata['folder'], DCI['importance_matrix'])
    DCI_disentanglement = DCI['disentanglement']
    
    bottleneck_lfcbm_dim = None
    if model_metadata['dataset'] in ['shapes3d','shapes3d_lfcbm']:
        bottleneck_lfcbm_dim = shapes3d_concepts_set_dim
    elif model_metadata['dataset'] in ['celeba','celeba_lfcbm']:
        bottleneck_lfcbm_dim = 99

    metrics_dict = {'DCI':DCI, 
                    'DCI_disentanglement': DCI_disentanglement,
                    'CE_concepts': None,
                    'CE_labels': np.mean(losses),
                    'concept_report': None,
                    'label_report': labels_report,
                    'lfcbm_bottleneckSize': bottleneck_lfcbm_dim,
                    'num_C': num_C
                    }
    
    activation_encoder = torch.load(os.path.join(model_metadata['activation_dir'], f'{model_metadata["dataset"]}_val_clip_RN50.pt'))

    print("Creating CS-LKG dataset with original concept set size: ", bottleneck_lfcbm_dim)
    create_CS_LKG_dataset({'all_concepts_predictions':all_concepts_predictions, 
                       'all_labels':all_labels, 
                       'all_encoder':activation_encoder,
                       'concept_loss':concept_loss,
                       'lfcbm_bottleneckSize': bottleneck_lfcbm_dim,
                       }, model_metadata['folder'], args)
    pickle.dump(metrics_dict, open(os.path.join(model_metadata['folder'], 'metrics.dict'), 'wb'))
    
    return metrics_dict, LaBo_metrics


    

def run_analysis_lf(model_metadata):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    activation_dir = model_metadata['activation_dir']
    base_folder = model_metadata['folder']

    # Load activation encoder afterwards in evaluate_lf
    #activation_encoder = torch.load(os.path.join(activation_dir, "shapes3d_val_clip_RN50.pt"))

    dataset = model_metadata['dataset'] 

    activation_image = torch.load(os.path.join(activation_dir, f"{dataset}_val_clip_ViT-B16.pt"))
    activation_image /= torch.norm(activation_image, dim=1, keepdim=True)
    activation_image = activation_image.to(device)
    activation_text = torch.load(os.path.join(activation_dir, f"{dataset}_handmade_concepts_ViT-B16.pt"))
    activation_text /= torch.norm(activation_text, dim=1, keepdim=True)
    activation_text = activation_text.to(device)
    P = activation_image @ activation_text.T

    index_to_remove = pickle.load(open(os.path.join(base_folder, "removed_concepts_id.list"), 'rb'))
    index_to_remove = list(set(index_to_remove))    # Remove duplicates
    print('Number of concepts removed: ', len(index_to_remove))
    print(P.shape[1])
    print('list removed ids:',index_to_remove)
    mask = torch.ones(P.shape[1], dtype=bool)
    mask[index_to_remove] = False
    model_metadata['mask'] = mask
    concs = sum(mask == 1)
    print('Number of concepts in the mask: ', concs, model_metadata['folder'])
    P = P[:, mask]
    
     
    if model_metadata['dataset'] in['shapes3d','celeba']:
        dataset_train = get_data(model_metadata['dataset'] + "_train")
        dataset_val = get_data(model_metadata['dataset'] + "_val")
    else:
        raise NotImplementedError(f'Dataset not implemented for {dataset}')
    print(len(dataset_val))

    args = pickle.load(open(os.path.join(base_folder, "args.pkl"), 'rb'))
                                                            
    if os.path.exists(os.path.join(base_folder, 'metrics.dict')) and os.path.exists(os.path.join(base_folder, 'LaBo/labo_metrics.dict')) and os.path.exists(os.path.join(base_folder, 'CS_dataset.ds')) and os.path.exists(os.path.join(base_folder, 'LKG_dataset.ds')):
        print('Metrics already computed for model, loading...')
        metrics = pickle.load(open(os.path.join(base_folder, 'metrics.dict'), 'rb'))
        LaBo_metrics = pickle.load(open(os.path.join(base_folder, 'LaBo/labo_metrics.dict'), 'rb'))
    else:
        # Dataloader initialization
        dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True)
        dl_test = torch.utils.data.DataLoader(dataset_val, batch_size=128, shuffle=False)
        # Model initialization
        model = load_cbm(base_folder, device = device)
        loss = torch.nn.CrossEntropyLoss()
        metrics, LaBo_metrics = evaluate_lf(model, dl_test, loss, device, model_metadata, args)

    if os.path.exists(os.path.join(base_folder, 'complete_analysis.dict')) and os.path.exists(os.path.join(base_folder, 'LaBo/complete_analysis.dict')):
        print('Metrics already computed for model, loading...')
        metrics = pickle.load(open(os.path.join(base_folder, 'complete_analysis.dict'), 'rb'))
        LaBo_metrics = pickle.load(open(os.path.join(base_folder, 'LaBo/complete_analysis.dict'), 'rb'))
    else:
        # Plug in different metadata in the evaluation function, so it understands it is using lfcbm 
        md = model_metadata
        md['dataset'] = model_metadata['dataset'] + '_lfcbm'
        md['version'] = str(metrics['num_C'])
        print(metrics)
        md['lfcbm_bottleneckSize'] = metrics['lfcbm_bottleneckSize']
        SUPERVISED_BOTTLENECK_DIM[md['dataset']][str(metrics['num_C'])] = metrics['num_C']
        
        setattr(args, 'num_C', metrics['num_C'])
        print(md['lfcbm_bottleneckSize'])
        leakage_results = leakage(md, args, n_epochs=50, already_removed_concepts=index_to_remove)
        CS_results = completeness_score(md, n_epochs=100)
        #
        LaBo_model_metadata = {'folder': os.path.join(base_folder,'LaBo'), 'version': md['version'], 'dataset': model_metadata['dataset']}
        LaBo_model_metadata['lfcbm_bottleneckSize'] = LaBo_metrics['lfcbm_bottleneckSize']
        print('LaBo indexes to remove: ', len(index_to_remove))
        LaBo_LKG = leakage(LaBo_model_metadata, args, n_epochs=50, already_removed_concepts=index_to_remove)
        LaBo_CS = completeness_score(LaBo_model_metadata, n_epochs=100)
        LaBo_metrics.update(LaBo_LKG)
        LaBo_metrics.update(LaBo_CS)
        #
        metrics.update(leakage_results)
        metrics.update(CS_results)
        metrics.update(model_metadata)
        pickle.dump(metrics, open(os.path.join(base_folder, 'complete_analysis.dict'), 'wb'))
        pickle.dump(LaBo_metrics, open(os.path.join(base_folder, 'LaBo/complete_analysis.dict'), 'wb'))
    
    #with open(os.path.join(model_metadata['folder'], 'concepts.txt')) as f:
    #    text_concetps = f.read().split("\n")
    #mask = np.ones(len(text_concetps), dtype=bool)
    #mask[index_to_remove] = False
    #text_concetps = [text_concetps[i] for i in range(len(text_concetps)) if mask[i]]
    # Save them as txt
    #with open(os.path.join(model_metadata['folder'], 'concepts_used_LKG_model.txt'), 'w') as f:
    #    for i,item in enumerate(text_concetps):
    #        f.write(f"{i}) {item}\n")


    return metrics, LaBo_metrics

def get_saved_models_to_run_lf(saved_models_lfcbm_folder):
    """
    Get the saved models to run analysis on
    """
    lf_cbm = {}
    lf_cbm_models = [f for f in os.listdir(saved_models_lfcbm_folder)]
    for v in lf_cbm_models:
          version = 'lfcbm'
          model_type = 'lfcbm'
          args = None
          checkpoint = False
          activations = False
          args_path = os.path.join(saved_models_lfcbm_folder,v,'args.pkl')
          dataset = v.split('_')[0]
          checkpoints_path = os.path.join(saved_models_lfcbm_folder,v)
          # Check if the args file is present
          if os.path.exists(args_path):
            args = args_path

          # Check if the checkpoint file is present
          if os.path.exists(os.path.join(saved_models_lfcbm_folder,v,'W_c.pt')) and os.path.exists(os.path.join(saved_models_lfcbm_folder,v,'W_g.pt')) and os.path.exists(os.path.join(saved_models_lfcbm_folder,v,'b_g.pt')):
              checkpoint = True
          # Check if the activations are present
          if len(os.listdir(os.path.join(saved_models_lfcbm_folder,v,'activations'))) == 5:
              activations = True

          # If all the necessary files are present, add the model to the list
          if args!=None and checkpoint and activations:
            lf_cbm[v] = {'dataset':dataset, 'model':model_type, 'version': version, 'args': args, 'activation_dir': os.path.join(checkpoints_path,'activations'), 'folder': checkpoints_path}
          else:
            print(f'Error: Missing files for model {v}')
            raise Exception(f'Error: Missing files for model {v}')
          
    
    return lf_cbm