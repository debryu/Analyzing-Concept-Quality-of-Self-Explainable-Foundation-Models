import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
import os
from torch.utils.data import DataLoader
from utils.estimators.entropy_estimators_nn import pY_W_evidence
from metrics.completeness import custom_CS
import pickle
from utils.preprocessing_model_data import INDEXES_TO_REMOVE, update_indices

# When training for leakage, this is the dimension of the bottleneck minus the concepts that are removed
# Currently not used
USABLE_BOTTLENECK_LKG = {
    'shapes3d': 28,
    'celeba': 28,
}

SUPERVISED_BOTTLENECK_DIM = {
    'shapes3d':{
        'fullsup': 42,
        'badsup': 40,
        '12sup': 12,
        '30sup': 30,
        'nolabel': 42,
    },

    'celeba':{
        'fullsup': 39,
        'badsup': 33,
        '12sup': 9, #12sup is actually 9sup because the celeba dataset has 3 less concepts than shapes3d
        '30sup': 30,    
        'nolabel': 39,
    },

    'shapes3d_lfcbm':{
        'lfcbm': None,  # This gets updated in the code depending on the concepts removed
    },
    
    'celeba_lfcbm':{
        'lfcbm': None,  # This gets updated in the code depending on the concepts removed
    }

}


class CS_estimator_Dataset(torch.utils.data.Dataset):
    def __init__(self, concepts, labels, encoders, args=None):
        self.args = args
        self.concepts = concepts
        self.labels = labels
        self.args = args
        self.encoders = encoders

    def __getitem__(self, idx):
        concepts = self.concepts[idx]
        labels = self.labels[idx]
        encoders = self.encoders[idx]
        return labels, concepts, encoders

    def __len__(self):
        return len(self.labels)

class Leakage_Dataset(torch.utils.data.Dataset):
    def __init__(self, concepts, labels, args=None):
        self.args = args
        self.concepts = concepts
        self.labels = labels

    def __getitem__(self, idx):
        concepts = self.concepts[idx]
        labels = self.labels[idx]
        return labels, concepts

    def __len__(self):
        return len(self.labels)
    


def train_random_predictor(model, dl, loss, probs, device = 'cuda', lr = 0.001):
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_losses = []
    probs = 1/probs
    prob_1 = probs[1]
    print('Training random predictor')
    for i, (_, concepts,_) in enumerate(tqdm(dl)):
        # Generate samples from bernoulli distribution with probability prob_1
        t = torch.ones(concepts.shape[0], 1).to(device)
        t = t*prob_1
        labels = torch.bernoulli(t)
        # Convert to one hot encoding
        labels = labels.view(-1).long()
        labels = torch.nn.functional.one_hot(labels, num_classes=2)
        #print(labels)
        labels = labels.to(device, torch.float)
        #print(labels)
        #print(labels.shape)
        concepts = concepts.to(device)
        preds = model(concepts)
        
        loss_val = torch.mean(loss(preds, labels))
        train_losses.append(loss_val.item())
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        
    
    return sum(train_losses)/len(train_losses)

def train_CS_predictor(model, dl, loss, device = 'cuda', lr = 0.001):
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_losses = []
    for i, (labels, concepts, encoders) in enumerate(tqdm(dl)):
        encoders = encoders.to(device)
        labels = labels.to(device, torch.long)
        concepts = concepts.to(device)
        #print(model.sup_type)
        if model.sup_type == 'badsup' and model.dataset == 'shapes3d':
            indexes_to_remove = [20,41]  # Create a tensor with all 1 except for the masked concepts
        elif model.sup_type == '12sup' and model.dataset == 'shapes3d':
            indexes_to_remove = [i for i in range(0,30)]
        elif model.sup_type == '30sup' and model.dataset == 'shapes3d':
            indexes_to_remove = [i for i in range(30,42)]
        elif model.sup_type == '12sup' and model.dataset == 'celeba':
            indexes_to_remove = [i for i in range(0,30)]
        elif model.sup_type == '30sup' and model.dataset == 'celeba':
            indexes_to_remove = [i for i in range(30,39)]
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
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        
    
    return sum(train_losses)/len(train_losses)


def test_CS_predictor(model, dl, loss, device = 'cuda'):
    model.eval()
    model.to(device)
    test_losses = []
    all_labels_predictions = []
    all_labels = []
    labels_report = None
    for i, (labels, concepts, encoders) in enumerate(dl):
        labels = labels.to(device, torch.long)
        concepts = concepts.to(device)
        if model.sup_type == 'badsup' and model.dataset == 'shapes3d':
            indexes_to_remove = [20,41]  # Create a tensor with all 1 except for the masked concepts
        elif model.sup_type == '12sup' and model.dataset == 'shapes3d':
            indexes_to_remove = [i for i in range(0,30)]
        elif model.sup_type == '30sup' and model.dataset == 'shapes3d':
            indexes_to_remove = [i for i in range(30,42)]
        elif model.sup_type == '12sup' and model.dataset == 'celeba':
            indexes_to_remove = [i for i in range(0,30)]
        elif model.sup_type == '30sup' and model.dataset == 'celeba':
            indexes_to_remove = [i for i in range(30,39)]
        else:
            indexes_to_remove = []
        concepts_mask = torch.ones(concepts.shape[1], dtype=torch.bool)
        concepts_mask[indexes_to_remove] = False
        encoders = encoders.to(device)
        if model.evidence_name == 'E':
            preds = model(encoders)
        else:
            preds = model(concepts[:,concepts_mask])
            all_labels.append(labels)
            all_labels_predictions.append(preds)

        # add here a classification report
        loss_val = torch.mean(loss(preds, labels))
        test_losses.append(loss_val.item())

    if model.evidence_name != 'E':
        all_labels = torch.cat(all_labels).cpu()  
        all_labels_predictions = torch.cat(all_labels_predictions).cpu()
        labels_pred_list = all_labels_predictions.argmax(axis = 1).tolist()
        labels_gt_list = all_labels.tolist()
        labels_report = classification_report(labels_gt_list, labels_pred_list, target_names = ['label_0', 'label_1'], output_dict=True)

    return sum(test_losses)/len(test_losses), labels_report

def train_LKG_predictor(model, dl, loss, device = 'cuda', lr = 0.001):
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_losses = []
    for i, (labels, concepts) in enumerate(tqdm(dl)):
        labels = labels.to(device, torch.long)
        concepts = concepts.to(device)
        preds = model(concepts)
        loss_val = torch.mean(loss(preds, labels))
        train_losses.append(loss_val.item())
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        
    return sum(train_losses)/len(train_losses)

def test_LKG_predictor(model, dl, loss, device = 'cuda'):
    model.eval()
    model.to(device)
    test_losses = []
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

    all_labels = torch.cat(all_labels).cpu()  
    all_labels_predictions = torch.cat(all_labels_predictions).cpu()
    #all_labels_predictions = (all_labels_predictions> 0.5).astype(int).tolist()
    labels_pred_list = all_labels_predictions.argmax(axis = 1).tolist()
    labels_gt_list = all_labels.tolist()
    

    labels_report = classification_report(labels_gt_list, labels_pred_list, target_names = ['label_0', 'label_1'], output_dict=True)
    #print(labels_report)
    return sum(test_losses)/len(test_losses), labels_report

def leakage(model_metadata, args, n_epochs = 10, batch_size = 128, lr = 0.001, already_removed_concepts = [], use_weighted_loss = False):
    print("Leakage", model_metadata)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(os.path.join(model_metadata['folder'],'LKG_dataset.ds'))
    train_dl = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(data['test'], batch_size=batch_size, shuffle=False)

    if 'ds_freq' in model_metadata.keys():
        freq_dict = model_metadata['ds_freq']
        CE_weight_labels = freq_dict['CE_weight_labels']
    else:
        CE_weight_labels = torch.ones(2)

    #indexes_to_remove = INDEXES_TO_REMOVE[args.dataset] # This is the list of concepts to remove
    # w.r.t. the original concept list
    # We need to update the indexes to match the concept list used by the model (after clip_cutoff and interpretability_cutoff)
    indexes_to_remove = INDEXES_TO_REMOVE[model_metadata['dataset']] # This is the list of concepts to remove
    #indexes_to_remove = update_indices(indexes_to_remove, already_removed_concepts)
    print('indexes_not_removed', indexes_to_remove)   
    print('already_removed_concepts', already_removed_concepts)
    print(update_indices(indexes_to_remove, already_removed_concepts))
    #indexes_to_remove = update_indices(indexes_to_remove, already_removed_concepts)
    indexes_to_remove = list(indexes_to_remove) + list(already_removed_concepts)
    indexes_to_remove = list(set(indexes_to_remove))          # Prevent duplicates
    # TENTATIVE
    print('indexes_to_remove', indexes_to_remove, len(indexes_to_remove))
    print('num_c:',args.num_C)
    #print(model_metadata['lfcbm_bottleneckSize'])
    print(len(indexes_to_remove))
    if 'lfcbm_bottleneckSize' in model_metadata.keys():
        print(' bottleneckSize:', model_metadata['lfcbm_bottleneckSize'])
        evidence_dim = model_metadata['lfcbm_bottleneckSize'] - len(indexes_to_remove)
    else:
        print('lfcbm_bottleneckSize not in model_metadata, using default')
        evidence_dim = args.num_C - len(indexes_to_remove)
    print(evidence_dim)
    P_Y_given_C = pY_W_evidence(evidence_dim=evidence_dim, Y_dim=2, n_layers=3, evidence_name='C', hidden_dim=1000, linear=False)
    if use_weighted_loss:
            loss = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.tensor(CE_weight_labels).to(device))
    else:
        loss = torch.nn.CrossEntropyLoss(reduction='none')
    
    min_test_loss = 1000
    best_report = None
    best_model_state_dict = None
    patience = 6
    for epoch in range(n_epochs):
        patience -= 1
        epoch_loss = train_LKG_predictor(P_Y_given_C,train_dl,loss,device,lr) 
        print(f'Epoch loss: {epoch_loss}')
        test_loss, labels_report = test_LKG_predictor(P_Y_given_C,test_dl,loss,device)
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            best_report = labels_report
            best_model_state_dict = P_Y_given_C.state_dict()
            patience = 6
        print(f'Test loss: {test_loss}')
        if test_loss < 1e-10 or patience == 0:
            break
    f1_score = best_report['macro avg']['f1-score']
    intepretable_score = abs(2*f1_score-1)

    ''' ALSO COMPUTE THE LAKEAGE FOR A LINEAR MODEL'''
    P_Y_given_C = pY_W_evidence(evidence_dim=evidence_dim, Y_dim=2, n_layers=3, evidence_name='C', hidden_dim=1000, linear=True)
    if use_weighted_loss:
            loss = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.tensor(CE_weight_labels).to(device))
    else:
        loss = torch.nn.CrossEntropyLoss(reduction='none')
    
    min_test_loss_l = 1000
    best_report_l = None
    best_model_state_dict_l = None
    patience = 6
    for epoch in range(n_epochs):
        patience -= 1
        epoch_loss = train_LKG_predictor(P_Y_given_C,train_dl,loss,device,lr) 
        print(f'Epoch loss: {epoch_loss}')
        test_loss, labels_report = test_LKG_predictor(P_Y_given_C,test_dl,loss,device)
        if test_loss < min_test_loss_l:
            min_test_loss_l = test_loss
            best_report_l = labels_report
            best_model_state_dict_l = P_Y_given_C.state_dict()
            patience = 6
        print(f'Test loss: {test_loss}')
        if test_loss < 1e-10 or patience == 0:
            break
    f1_score_l = best_report_l['macro avg']['f1-score']
    intepretable_score_l = abs(2*f1_score_l-1)
    ''' END OF LINEAR MODEL COMPUTATION '''

    leakage_results = {'best_report': best_report, 'CE_labels': min_test_loss, 'f1-score-leakage': f1_score, 'LKG_interpretable_score': intepretable_score,
                       'best_report_linear': best_report_l, 'CE_labels_linear': min_test_loss_l, 'f1-score-leakage_linear': f1_score_l, 'LKG_interpretable_score_linear': intepretable_score_l}
    
    pickle.dump(leakage_results, open(os.path.join(model_metadata['folder'],'LKG_results.dict'), 'wb'))
    torch.save(best_model_state_dict, os.path.join(model_metadata['folder'],'leakage_predictor.statedict'))
    torch.save(best_model_state_dict_l, os.path.join(model_metadata['folder'],'leakage_linear_predictor.statedict'))
    return leakage_results
    

def completeness_score(model_metadata, n_epochs = 10, batch_size = 128, lr = 0.001, use_weighted_loss = False):
    print("Completeness score", model_metadata)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(os.path.join(model_metadata['folder'],'CS_dataset.ds'))
    # data is a class that has self.args variable which is a dictionary
    H_Y = data['test'].args['H_Y']
    CE_concepts = data['test'].args['CE_concepts']
    frequency = data['test'].args['frequency']
    ds_used = model_metadata['dataset']
    if 'ds_freq' in model_metadata.keys():
        freq_dict = model_metadata['ds_freq']
        CE_weight_labels = freq_dict['CE_weight_labels']
    else:
        if ds_used.endswith('lfcbm'):
            ds_used = ds_used.split('_')[0]
        if os.path.exists(f'data/ds_freqs/{ds_used}_freq.pkl'):
            ds_freq = pickle.load(open(f'data/ds_freqs/{ds_used}_freq.pkl', 'rb'))
            CE_weight = ds_freq['CE_weight']
            frequencies = ds_freq['frequencies']
            CE_weight_labels = ds_freq['CE_weight_labels']
        else:
            raise NotImplementedError()
    

    _,_,encoder = data['train'][0]   # Get the first sample of train, the encoders_dim is the same for all samples (train and test)
    encoders_dim = encoder.shape[0]
    train_dl = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(data['test'], batch_size=batch_size, shuffle=False)
    if model_metadata['dataset'] not in SUPERVISED_BOTTLENECK_DIM.keys():
        raise NotImplementedError(f"The dataset {model_metadata['dataset']} is not implemented in utils.estimators.estimator_model")
    supervised_bottleneck_dim = SUPERVISED_BOTTLENECK_DIM[model_metadata['dataset']]
    print(supervised_bottleneck_dim)
    if model_metadata['version'] not in supervised_bottleneck_dim.keys():
        raise NotImplementedError(f"The version {model_metadata['version']} is not implemented in utils.estimators.estimator_model")
    supervised_bottleneck_dim = supervised_bottleneck_dim[model_metadata['version']]
   
    P_Y_given_C = pY_W_evidence(evidence_dim=supervised_bottleneck_dim, Y_dim=2, n_layers=3, 
                                evidence_name='C', sup_type=model_metadata['version'], 
                                dataset= model_metadata['dataset'], hidden_dim=1000)
    P_Y_given_E = pY_W_evidence(evidence_dim=encoders_dim, Y_dim=2, n_layers=3, evidence_name='E', hidden_dim=1000)
    models = [P_Y_given_C, P_Y_given_E]
    best_models = []
    best_report = None
    for i,model in enumerate(models):
        if use_weighted_loss:
            loss = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.tensor(CE_weight_labels).to(device))
        else:
            loss = torch.nn.CrossEntropyLoss(reduction='none')
        
        min_test_loss = 1000
        best_model_state_dict = None
        patience = 6
        for epoch in range(n_epochs):
            patience -= 1
            epoch_loss = train_CS_predictor(model,train_dl,loss,device,lr) 
            print(f'Epoch loss CS classifier: {epoch_loss}')
            test_loss, label_report = test_CS_predictor(model,test_dl,loss,device)
            if test_loss < min_test_loss:
                min_test_loss = test_loss
                best_model_state_dict = model.state_dict()
                patience = 6
                if i == 0:
                    best_report = label_report
            print(f'Test loss: {test_loss}')
            if test_loss < 1e-10 or patience == 0:
                break
        best_models.append({'state_dict': best_model_state_dict, 'test_loss': min_test_loss})

    
    torch.save(best_models[0]['state_dict'], os.path.join(model_metadata['folder'],'P_Y_given_C.statedict'))
    entropy_given_C = best_models[0]['test_loss']

    torch.save(best_models[1]['state_dict'], os.path.join(model_metadata['folder'],'P_Y_given_E.statedict'))
    entropy_given_E = best_models[1]['test_loss']


    CS, WORSTCASE = custom_CS(H_Y, entropy_given_C, entropy_given_E, CE_concepts)
    
    '''   RANDOM PREDICTOR    '''
    # Train random predictor for the leakage estimation
    #ds_to_use = model_metadata['dataset']
    #print('ds_to_use:', ds_to_use)
    #asd
    if model_metadata['dataset'] in ['shapes3d']:
        ed = 42
    elif model_metadata['dataset'] in ['celeba']:
        ed = 39
    elif model_metadata['dataset'] in ['celeba_lfcbm','shapes3d_lfcbm']:
        ed = supervised_bottleneck_dim
    else:
        raise NotImplementedError(f"The dataset {model_metadata['dataset']} is not implemented in utils.estimators.estimator_model")

    random_predictor_model = pY_W_evidence(evidence_dim=ed, Y_dim=2, n_layers=3, 
                                evidence_name='C', sup_type=model_metadata['version'], 
                                dataset= model_metadata['dataset'], hidden_dim=1000)
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    min_test_loss = 1000
    best_model_state_dict = None
    probs = torch.tensor(CE_weight_labels).to(device)
    patience = 10
    for epoch in range(n_epochs):
        patience -= 1
        epoch_loss = train_random_predictor(random_predictor_model,train_dl,loss,probs,device,lr) 
        print(f'Epoch loss CS classifier: {epoch_loss}')
        if epoch_loss < min_test_loss:
            min_test_loss = epoch_loss
            
        if test_loss < 1e-10 or patience == 0:
            break

    CE_random_predictor = min_test_loss
    print('Random predictor test loss:', min_test_loss)
    
    '''   RANDOM PREDICTOR    '''

    
    res = {'CS': CS, 'WORSTCASE': WORSTCASE, 'H_Y': H_Y, 'CE_concepts_test_split': CE_concepts, 
            'frequency_test_split': frequency, 'loss_Y_given_C': best_models[0]['test_loss'], 
            'loss_Y_given_E': best_models[1]['test_loss'], 'best_report_Y_given_C': best_report, 
            'CE_random_predictor': CE_random_predictor}
    pickle.dump(res, open(os.path.join(model_metadata['folder'],'CS_results.dict'), 'wb'))
    return res