import torch
import os
from torch.utils.data import DataLoader
from random import choice
from tqdm import tqdm 
from metrics.completeness import custom_CS
from utils.estimators.entropy_estimators_nn import pY_W_evidence
from utils.utils import CS_estimator_Dataset
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import math
import pickle

'''
n_epochs = 10
skip_models = []
skip_versions = []
skip_seeds = []
skip_betas = []

DO_LEAKAGE = True
DO_CS = False
'''

def create_folder(dataset, model, version, beta, seed, path = 'data/ckpt/saved_models/evaluated'):
  '''
    Create the folder to store the data
  '''
  
  folder = path + f'/{dataset}_{model}_{version}_{beta}_{seed}'
  if not os.path.exists(folder):
    os.makedirs(folder)
  return folder

def train_predictor(model, dl, loss, device = 'cuda', lr = 0.001):
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_losses = []
    train_log_losses = []
    for i, (labels, concepts, encoders) in enumerate(tqdm(dl)):
        labels = labels.to(device, torch.long)
        concepts = concepts.to(device)
        #print(model.sup_type)
        if model.sup_type == 'badsup':
            indexes_to_remove = [20,41]  # Create a tensor with all 1 except for the masked concepts
        elif model.sup_type == '12sup':
            indexes_to_remove = [i for i in range(0,30)]
        elif model.sup_type == '30sup':
            indexes_to_remove = [i for i in range(30,42)]
        else:
            indexes_to_remove = []
        
        concepts_mask = torch.ones(concepts.shape[1], dtype=torch.bool)
        concepts_mask[indexes_to_remove] = False
        encoders = encoders.to(device)
        if model.evidence_name == 'E':
            preds = model(encoders)
        else:
            preds = model(concepts[:,concepts_mask])
        
        loss_val = torch.mean(loss(preds, labels))
        train_losses.append(loss_val.item())
        loss_values = torch.log2(loss(preds, labels).detach())
        train_log_losses.append(loss_values)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        
    
    train_log_losses = torch.cat(train_log_losses).cpu().numpy()
    return sum(train_losses)/len(train_losses), sum(train_log_losses)/len(train_log_losses)


def test_predictor(model, dl, loss, device = 'cuda'):
    model.eval()
    model.to(device)
    test_losses = []
    test_log_losses = []
    for i, (labels, concepts, encoders) in enumerate(dl):
        labels = labels.to(device, torch.long)
        concepts = concepts.to(device)
        if model.sup_type == 'badsup':
            indexes_to_remove = [20,41]  # Create a tensor with all 1 except for the masked concepts
        elif model.sup_type == '12sup':
            indexes_to_remove = [i for i in range(0,30)]
        elif model.sup_type == '30sup':
            indexes_to_remove = [i for i in range(30,42)]
        else:
            indexes_to_remove = []
        concepts_mask = torch.ones(concepts.shape[1], dtype=torch.bool)
        concepts_mask[indexes_to_remove] = False
        encoders = encoders.to(device)
        if model.evidence_name == 'E':
            preds = model(encoders)
        else:
            preds = model(concepts[:,concepts_mask])

        loss_val = torch.mean(loss(preds, labels))
        test_losses.append(loss_val.item())
        loss_values = torch.log2(loss(preds, labels).detach())
        test_log_losses.append(loss_values)

    test_log_losses = torch.cat(test_log_losses).cpu().numpy()
    return sum(test_losses)/len(test_losses), sum(test_log_losses)/len(test_log_losses)

def train_lkg_predictor(model, dl, loss, device = 'cuda', lr = 0.001):
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_losses = []
    train_log_losses = []
    for i, (labels, concepts) in enumerate(tqdm(dl)):
        labels = labels.to(device, torch.long)
        concepts = concepts.to(device)
        preds = model(concepts)
        
        loss_val = torch.mean(loss(preds, labels))
        train_losses.append(loss_val.item())
        loss_values = torch.log2(loss(preds, labels).detach())
        train_log_losses.append(loss_values)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        
    
    train_log_losses = torch.cat(train_log_losses).cpu().numpy()
    return sum(train_losses)/len(train_losses), sum(train_log_losses)/len(train_log_losses)

def test_lkg_predictor(model, dl, loss, device = 'cuda'):
    model.eval()
    model.to(device)
    test_losses = []
    test_log_losses = []
    all_labels_predictions = []
    all_labels = []
    for i, (labels, concepts) in enumerate(dl):
        labels = labels.to(device, torch.long)
        concepts = concepts.to(device)
        preds = model(concepts)

        all_labels.append(labels)
        all_labels_predictions.append(preds)
        loss_val = torch.mean(loss(preds, labels))
        test_losses.append(loss_val.item())
        loss_values = torch.log2(loss(preds, labels).detach())
        test_log_losses.append(loss_values)

    test_log_losses = torch.cat(test_log_losses).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu()  
    all_labels_predictions = torch.cat(all_labels_predictions).cpu()
    #all_labels_predictions = (all_labels_predictions> 0.5).astype(int).tolist()
    labels_pred_list = all_labels_predictions.argmax(axis = 1).tolist()
    labels_gt_list = all_labels.tolist()
    

    labels_report = classification_report(labels_gt_list, labels_pred_list, target_names = ['label_0', 'label_1'], output_dict=True)
    print(labels_report)
    return sum(test_losses)/len(test_losses), labels_report


def leakage_completeness_score(n_epochs = 10,
                               skip_models = [],
                               skip_versions=[],
                               skip_seeds=[],
                               skip_betas=[],
                               args = None,
                               DO_LEAKAGE = True, DO_CS = True):
    results_dict = {}
    lr = 0.001
    batch_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = os.path.dirname(os.path.realpath(__file__))
    #path = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ML/Tirocinio/interpreter'
    models_path = '/data/encoder_dataset/'

    results = []
    for file in os.listdir(path + models_path):
        # LEAKAGE score
        '''
        -------------------------------------------------------
                LEAKAGE SCORE
        -------------------------------------------------------
        
        '''
        if file.endswith('.pth') and file.startswith('LKG') and DO_LEAKAGE:
            print(f'Training for LEAKAGE: {file}')
            #print(file)
            
            '''
                # Manually inspect the predictions
            if version != 'labelsupervision':
            continue
            '''
            dataset_type = file.split('_')[0]     # Example CS or LKG
            # model_type added later
            if len(file.split('_')) == 5:
                version = file.split('_')[1]       # Example betaglancenet
                model_type = file.split('_')[2]          # Example badsup
                seed = file.split('_')[3]             # Example beta1.0.pt
                beta = file.split('_')[4]             # Ecample seed90
                beta = beta.split('.pt')[0]
            else:
                model_type = None
                version = file.split('_')[1]          # Example badsup
                seed = file.split('_')[2]             # Example beta1.0.pt
                beta = file.split('_')[3]             # Ecample seed90
                beta = beta.split('.pt')[0]
            
            if model_type not in results_dict:
                results_dict[model_type] = {}
            if version not in results_dict[model_type]:
                results_dict[model_type][version] = {}
            if beta not in results_dict[model_type][version]:
                results_dict[model_type][version][beta] = {}
            if seed not in results_dict[model_type][version][beta]:
                results_dict[model_type][version][beta][seed] = {}
        

            # Enable skipping
            if seed in skip_seeds:
                print(f'Skipping {seed}.')
                continue
            if model_type in skip_models:
                print(f'Skipping {version}.')
                continue
            if version in skip_versions:
                print(f'Skipping {version}.')
                continue
            if beta in skip_betas:
                print(f'Skipping {beta}.')
                continue
            

            # Open the experinent folder
            experiment_folder = create_folder(args.dataset, model_type, version, beta, seed)
            if not os.path.exists(f'{experiment_folder}/LKG_dataset.pth'):
                print('Trying to open the file: ', f'{experiment_folder}/LKG_dataset.pth')
                raise ValueError('The file does not exists!')
            data = torch.load(f'{experiment_folder}/LKG_dataset.pth')
            #H_Y = data['test'].args['H_Y']
            #CE_concepts = data['test'].args['CE_concepts']
            #_,_,encoders_dim = data['train'][0]
            #encoders_dim = encoders_dim.shape[0]
            train_dl = DataLoader(data['train'], batch_size=batch_size, shuffle=False)
            test_dl = DataLoader(data['test'], batch_size=batch_size, shuffle=False)
            P_Y_given_C = pY_W_evidence(evidence_dim=28, Y_dim=2, n_layers=3, evidence_name='C', hidden_dim=1000, linear=False)
            
            models = [P_Y_given_C]
            min_loss = []
            for model in models:
                model.train()
                loss = torch.nn.CrossEntropyLoss(reduction='none')
                
                min_test_loss = 1000
                best_report = None
                patience = 6
                for epoch in range(n_epochs):
                    patience -= 1
                    epoch_loss, train_log_loss = train_lkg_predictor(model,train_dl,loss,device,lr) 
                    print(f'Epoch loss: {epoch_loss}')
                    test_loss, labels_report = test_lkg_predictor(model,test_dl,loss,device)
                    if test_loss < min_test_loss:
                        min_test_loss = test_loss
                        best_report = labels_report
                        patience = 6
                    print(f'Test loss: {test_loss}')
                    if test_loss < 1e-10 or patience == 0:
                        break

                #print(model.hidden_fc[0].state_dict()['weight'])
                    
                min_loss.append(min_test_loss)

            f1_score = best_report['macro avg']['f1-score']
            res = {'CE_labels': min_loss[0], 'f1-score-leakage': f1_score}
            # Create a folder to store the results
            experiment_folder = create_folder(args.dataset, model_type, version, beta, seed)
            # Make sure there is no overlapping
            if os.path.exists(experiment_folder + '/metrics_leakage.pkl'):
                print('Trying to create the file: ', experiment_folder + '/metrics_leakage.pkl')
                raise ValueError('The file already exists!')
            pickle.dump(res, open(experiment_folder + '/metrics_leakage.pkl', 'wb'))
            if model_type != None:
                results_dict[model_type][version][beta][seed].update(res)
                results.append(f'Model {version} {model_type} {beta} {seed} | CE labels: {min_loss[0]} | f1-score {f1_score} | \n{best_report}\n')
            else:
                print('ERROR')
                results.append(f'Model {version} {beta} {seed} | CE labels: {min_loss[0]} | f1-score {f1_score} | \n{best_report}\n')

        # COMPLETENESS SCORE     
        '''
        -------------------------------------------------------
                COMPLETENESS SCORE      +++++++++++++++++++++++
        -------------------------------------------------------
        
        '''   
        if file.endswith('.pth') and file.startswith('CS') and DO_CS:
            print(f'Training for CS: {file}')
            #print(file)
            
            '''
                # Manually inspect the predictions
            if version != 'labelsupervision':
            continue
            '''
            dataset_type = file.split('_')[0]     # Example CS or LKG
            # model_type added later
            if len(file.split('_')) == 5:
                version = file.split('_')[1]       # Example betaglancenet
                model_type = file.split('_')[2]          # Example badsup
                seed = file.split('_')[3]             # Example beta1.0.pt
                beta = file.split('_')[4]             # Ecample seed90
                beta = beta.split('.pt')[0]
            else:
                model_type = None
                version = file.split('_')[1]          # Example badsup
                seed = file.split('_')[2]             # Example beta1.0.pt
                beta = file.split('_')[3]             # Ecample seed90
                beta = beta.split('.pt')[0]
                
            if model_type not in results_dict:
                results_dict[model_type] = {}
            if version not in results_dict[model_type]:
                results_dict[model_type][version] = {}
            if beta not in results_dict[model_type][version]:
                results_dict[model_type][version][beta] = {}
            if seed not in results_dict[model_type][version][beta]:
                results_dict[model_type][version][beta][seed] = {}

            # Enable skipping
            if seed in skip_seeds:
                print(f'Skipping {seed}.')
                continue
            if model_type in skip_models:
                print(f'Skipping {version}.')
                continue
            if version in skip_versions:
                print(f'Skipping {version}.')
                continue
            if beta in skip_betas:
                print(f'Skipping {beta}.')
                continue
            
            # Open the experinent folder
            experiment_folder = create_folder(args.dataset, model_type, version, beta, seed)
            if not os.path.exists(f'{experiment_folder}/CS_dataset.pth'):
                print('Trying to open the file: ', f'{experiment_folder}/CS_dataset.pth')
                raise ValueError('The file does not exists!')
            data = torch.load(f'{experiment_folder}/CS_dataset.pth')

            H_Y = data['test'].args['H_Y']
            CE_concepts = data['test'].args['CE_concepts']
            _,_,encoders_dim = data['train'][0]
            encoders_dim = encoders_dim.shape[0]
            train_dl = DataLoader(data['train'], batch_size=batch_size, shuffle=False)
            test_dl = DataLoader(data['test'], batch_size=batch_size, shuffle=False)

            if version == 'badsup':
                supervised_bottleneck_dim = 40
            elif version == '12sup':
                supervised_bottleneck_dim = 12
            elif version == '30sup':
                supervised_bottleneck_dim = 30
            else:
                supervised_bottleneck_dim = 42

            P_Y_given_C = pY_W_evidence(evidence_dim=supervised_bottleneck_dim, Y_dim=2, n_layers=3, evidence_name='C', sup_type=version, hidden_dim=1000)
            P_Y_given_E = pY_W_evidence(evidence_dim=encoders_dim, Y_dim=2, n_layers=3, evidence_name='E', hidden_dim=1000)

            models = [P_Y_given_C, P_Y_given_E]
            approx = []
            for model in models:
                model.train()
                loss = torch.nn.CrossEntropyLoss(reduction='none')
                
                min_test_loss = 1000
                for epoch in range(n_epochs):
                    epoch_loss, train_log_loss = train_predictor(model,train_dl,loss,device,lr) 
                    print(f'Epoch loss: {epoch_loss}')
                    test_loss, test_log_loss = test_predictor(model,test_dl,loss,device)
                    if test_loss < min_test_loss:
                        min_test_loss = test_loss
                    print(f'Test loss: {test_loss}, Test log loss: {test_log_loss}')
                    if test_loss < 1e-10:
                        break
                approx.append(min_test_loss)
            

            CS, WORSTCASE = custom_CS(H_Y, approx[0], approx[1], CE_concepts)
            res = {'CS': CS, 'WORSTCASE': WORSTCASE}
            results_dict[model_type][version][beta][seed].update(res)

            # Create a folder to store the results
            experiment_folder = create_folder(args.dataset, model_type, version, beta, seed)
            # Make sure there is no overlapping
            if os.path.exists(experiment_folder + '/metrics_completeness_score.pkl'):
                print('Trying to create the file: ', experiment_folder + '/metrics_completeness_score.pkl')
                raise ValueError('The file already exists!')
            pickle.dump(res, open(experiment_folder + '/metrics_completeness_score.pkl', 'wb'))


            if model_type != None:
                results.append(f'Model {version} {model_type} {beta} {seed} | CS: {CS}  | Lower bound: {WORSTCASE}')
            else:
                results.append(f'Model {version} {beta} {seed} | CS: {CS}  | Lower bound: {WORSTCASE}')

    return results_dict

'''
print('---- STARTING ------')
lr = 0.001
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = os.path.dirname(os.path.realpath(__file__))
#path = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ML/Tirocinio/interpreter'
models_path = '/data/encoder_dataset/'

results = []
for file in os.listdir(path + models_path):
    # LEAKAGE score
    
#    -------------------------------------------------------
#            LEAKAGE SCORE
#    -------------------------------------------------------
   

    if file.endswith('.pth') and file.startswith('LKG') and DO_LEAKAGE:
        print(f'Training for LEAKAGE: {file}')
        #print(file)
        
        
              # Manually inspect the predictions
        #if version != 'labelsupervision':
        #   continue
        
        dataset_type = file.split('_')[0]     # Example CS or LKG
        # model_type added later
        if len(file.split('_')) == 5:
            version = file.split('_')[1]       # Example betaglancenet
            model_type = file.split('_')[2]          # Example badsup
            seed = file.split('_')[3]             # Example beta1.0.pt
            beta = file.split('_')[4]             # Ecample seed90
            beta = beta.split('.pt')[0]
        else:
            model_type = None
            version = file.split('_')[1]          # Example badsup
            seed = file.split('_')[2]             # Example beta1.0.pt
            beta = file.split('_')[3]             # Ecample seed90
            beta = beta.split('.pt')[0]
        
        # Enable skipping
        if seed in skip_seeds:
            print(f'Skipping {seed}.')
            continue
        if model_type in skip_models:
            print(f'Skipping {version}.')
            continue
        if version in skip_versions:
            print(f'Skipping {version}.')
            continue
        if beta in skip_betas:
            print(f'Skipping {beta}.')
            continue
        
        data = torch.load(path + models_path + file)
        #H_Y = data['test'].args['H_Y']
        #CE_concepts = data['test'].args['CE_concepts']
        #_,_,encoders_dim = data['train'][0]
        #encoders_dim = encoders_dim.shape[0]
        train_dl = DataLoader(data['train'], batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(data['test'], batch_size=batch_size, shuffle=False)
        P_Y_given_C = pY_W_evidence(evidence_dim=28, Y_dim=2, n_layers=3, evidence_name='C', hidden_dim=1000, linear=True)
        
        models = [P_Y_given_C]
        min_loss = []
        for model in models:
            model.train()
            loss = torch.nn.CrossEntropyLoss(reduction='none')
            
            min_test_loss = 1000
            best_report = None
            for epoch in range(n_epochs):
              epoch_loss, train_log_loss = train_lkg_predictor(model,train_dl,loss,device,lr) 
              print(f'Epoch loss: {epoch_loss}')
              test_loss, labels_report = test_lkg_predictor(model,test_dl,loss,device)
              if test_loss < min_test_loss:
                  min_test_loss = test_loss
                  best_report = labels_report
              print(f'Test loss: {test_loss}')
              if test_loss < 1e-10:
                  break

            #print(model.hidden_fc[0].state_dict()['weight'])
                  
            min_loss.append(min_test_loss)
        

        if model_type != None:
            f1_score = best_report['macro avg']['f1-score']
            results.append(f'Model {version} {model_type} {beta} {seed} | CE labels: {min_loss[0]} | f1-score {f1_score} | \n{best_report}\n')
        else:
            results.append(f'Model {version} {beta} {seed} | CE labels: {min_loss[0]} | f1-score {f1_score} | \n{best_report}\n')

    # COMPLETENESS SCORE     
    
    #-------------------------------------------------------
    #        COMPLETENESS SCORE      +++++++++++++++++++++++
    #-------------------------------------------------------
    
    if file.endswith('.pth') and file.startswith('CS') and DO_CS:
        print(f'Training for CS: {file}')
        #print(file)
        
        
              # Manually inspect the predictions
        #if version != 'labelsupervision':
        #   continue
        
        dataset_type = file.split('_')[0]     # Example CS or LKG
        # model_type added later
        if len(file.split('_')) == 5:
            version = file.split('_')[1]       # Example betaglancenet
            model_type = file.split('_')[2]          # Example badsup
            seed = file.split('_')[3]             # Example beta1.0.pt
            beta = file.split('_')[4]             # Ecample seed90
            beta = beta.split('.pt')[0]
        else:
            model_type = None
            version = file.split('_')[1]          # Example badsup
            seed = file.split('_')[2]             # Example beta1.0.pt
            beta = file.split('_')[3]             # Ecample seed90
            beta = beta.split('.pt')[0]
        
        # Enable skipping
        if seed in skip_seeds:
            print(f'Skipping {seed}.')
            continue
        if model_type in skip_models:
            print(f'Skipping {version}.')
            continue
        if version in skip_versions:
            print(f'Skipping {version}.')
            continue
        if beta in skip_betas:
            print(f'Skipping {beta}.')
            continue
        
        data = torch.load(path + models_path + file)
        H_Y = data['test'].args['H_Y']
        CE_concepts = data['test'].args['CE_concepts']
        _,_,encoders_dim = data['train'][0]
        encoders_dim = encoders_dim.shape[0]
        train_dl = DataLoader(data['train'], batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(data['test'], batch_size=batch_size, shuffle=False)

        if version == 'badsup':
            supervised_bottleneck_dim = 40
        elif version == '12sup':
            supervised_bottleneck_dim = 12
        elif version == '30sup':
            supervised_bottleneck_dim = 30
        else:
            supervised_bottleneck_dim = 42

        P_Y_given_C = pY_W_evidence(evidence_dim=supervised_bottleneck_dim, Y_dim=2, n_layers=3, evidence_name='C', sup_type=version, hidden_dim=1000)
        P_Y_given_E = pY_W_evidence(evidence_dim=encoders_dim, Y_dim=2, n_layers=3, evidence_name='E', hidden_dim=1000)

        models = [P_Y_given_C, P_Y_given_E]
        approx = []
        for model in models:
            model.train()
            loss = torch.nn.CrossEntropyLoss(reduction='none')
            
            min_test_loss = 1000
            for epoch in range(n_epochs):
              epoch_loss, train_log_loss = train_predictor(model,train_dl,loss,device,lr) 
              print(f'Epoch loss: {epoch_loss}')
              test_loss, test_log_loss = test_predictor(model,test_dl,loss,device)
              if test_loss < min_test_loss:
                  min_test_loss = test_loss
              print(f'Test loss: {test_loss}, Test log loss: {test_log_loss}')
              if test_loss < 1e-10:
                  break
            approx.append(min_test_loss)
        

        CS, WORSTCASE = custom_CS(H_Y, approx[0], approx[1], CE_concepts)

        if model_type != None:
            results.append(f'Model {version} {model_type} {beta} {seed} | CS: {CS}  | Lower bound: {WORSTCASE}')
        else:
            results.append(f'Model {version} {beta} {seed} | CS: {CS}  | Lower bound: {WORSTCASE}')
          
              
print('---- TERMINATED ------')
print('\n\n\n')
for r in results:
    print(r)
'''