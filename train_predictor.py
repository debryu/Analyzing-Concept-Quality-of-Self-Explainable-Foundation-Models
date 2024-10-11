import torch
import os
from torch.utils.data import DataLoader
from random import choice
from tqdm import tqdm 
from utils.utils import Linear_predictor, Linear_predictor_multiconcepts
from utils.utils import Leakage_Dataset
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def train_test_multiconcept_predictor(n_concepts:int, train_dl, test_dl, device = 'cuda', lr = 0.001, epochs = 50):
  # Initialize model
  model = Linear_predictor_multiconcepts(n_concepts, 2, device=device)
  # Random initialization
  model.fc.weight.data = torch.randn(2, n_concepts).to(device)   # The weight matrix shape is (out_features, in_features)
  # Adjust the loss function to take into account the class imbalance
  loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1/(1-CLASS_1_FREQUENCY_IN_DS), 1/CLASS_1_FREQUENCY_IN_DS]).to(device))
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  # Train
  model.train() 
  train_losses = []
  for epoch in range(epochs):
    for i, (labels, concepts) in enumerate(train_dl):
        labels = labels.to(torch.long).to(device)
        concepts = concepts.to(device)
        preds = model(concepts[:,:n_concepts])
        loss_value = loss(preds, labels)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        train_losses.append(loss_value.item())
    print(f'Epoch {epoch} | Loss: {sum(train_losses)/len(train_losses)}')
    # Test
  model.eval()
  y_pred = []
  y_true = []
  for i, (labels, concepts) in enumerate(test_dl):
      labels = labels.to(torch.long).to(device)
      concepts = concepts.to(device)
      preds = model(concepts[:,:n_concepts])
      y_pred_list = preds.argmax(dim=1).tolist()

      for y in y_pred_list:
          y_pred.append(y)
      y_true_list = labels.tolist()
      for y in y_true_list:
          y_true.append(y)

      '''
      # Manually inspect the predictions
      if version == 'labelsupervision':
          if pair == 'informative':
              for i, lab in enumerate(labels.tolist()):
                print(i,lab,concepts[i,concept1_idx].item(), concepts[i,concept2_idx].item())      
      else:
        continue
      '''
  scores = classification_report(y_true, y_pred, output_dict=True)
  macro_avg = scores['macro avg']['precision']
  out_dict = {'macro_avg': macro_avg,'n_concepts': n_concepts}
  return out_dict

    


print('---- STARTING ------')
lr = 0.001
batch_size = 128
test_batch_size = 1024
epochs = 50
deterministic = True
CLASS_1_FREQUENCY_IN_DS = 0.03663
device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = os.path.dirname(os.path.realpath(__file__))
#path = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ML/Tirocinio/interpreter'
models_path = '/data/leakage_dataset/'

non_informative_concepts = [0,1,2,3,4,5,6,7,8,9,              # Floor Hue
                            10,11,12,13,14,15,16,17,18,19,    # Obj Hue
                            #20,21,22,23,24,25,26,27,28,29,    # Background Hue
                            30,31,32,33,34,35,36,37,          # Scale
                            #38,39,40,41                       # Shape
                            ]
if deterministic:
  pairs = {'informative': (20,41), 
          'partially_informative': (41, 2), 
          'non_informative_1': (15,19),
          'non_informative_2': (33,6),
          'non_informative_3': (14,8),
          'decoy': 'random'}
else:
  pairs = {'informative': (20,41), 
          'partially_informative': (41, choice(non_informative_concepts)), 
          'non_informative_1': (choice(non_informative_concepts),choice(non_informative_concepts)),
          'non_informative_2': (choice(non_informative_concepts),choice(non_informative_concepts)),
          'non_informative_3': (choice(non_informative_concepts),choice(non_informative_concepts)),
          'decoy': 'random'}
print(f'The pairs are: {pairs}')


results = []
for file in os.listdir(path + models_path):
    if file.endswith('.pth'):
        #print(file)
        version = file.split('_')[1]
        '''
              # Manually inspect the predictions
        if version != 'labelsupervision':
           continue
        '''
        seed = file.split('_')[2] 
        beta = file.split('_')[3]
        if seed not in ['seed14']:
            continue
        if beta not in ['beta1.pt.pth']:
            continue
        
        data = torch.load(path + models_path + file)
        train_dl = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(data['test'], batch_size=test_batch_size, shuffle=True)
        
        res_1 = []
        for i in range(2,43):
          res_1.append(train_test_multiconcept_predictor(i, train_dl, test_dl, device, lr, epochs))

        precisions = [res['macro_avg'] for res in res_1]
        plt.plot(range(2,43), precisions)
        plt.xlabel('Number of concepts')
        plt.ylabel('Macro avg precision')
        plt.title(f'Model {version} {seed} {beta}')
        plt.show()
        for res in res_1:
           print(f'Model {version} {beta} | Macro avg precision for {res["n_concepts"]} concepts: {res["macro_avg"]}')
           
        for pair in pairs:
          if pair != 'decoy':
            concept1_idx, concept2_idx = pairs[pair]

          # Initialize model
          model = Linear_predictor(2, 2, device=device)
          # Random initialization
          model.fc.weight.data = torch.randn(2, 2).to(device)
          # Adjust the loss function to take into account the class imbalance
          loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1/(1-CLASS_1_FREQUENCY_IN_DS), 1/CLASS_1_FREQUENCY_IN_DS]).to(device))
          optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

          # Train
          model.train() 
          train_losses = []
          for epoch in range(epochs):
            for i, (labels, concepts) in enumerate(train_dl):
                labels = labels.to(torch.long).to(device)
                concepts = concepts.to(device)
                if pair == 'decoy':
                  concept1 = torch.rand(concepts.size(0), 1).to(device).squeeze(1)
                  concept2 = torch.rand(concepts.size(0), 1).to(device).squeeze(1)
                  preds = model(concept1, concept2)
                else:
                  preds = model(concepts[:,concept1_idx], concepts[:,concept2_idx])
                loss_value = loss(preds, labels)
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
                train_losses.append(loss_value.item())
            print(f'Epoch {epoch} | Loss: {sum(train_losses)/len(train_losses)}')
                

          # Test
          model.eval()
          y_pred = []
          y_true = []
          for i, (labels, concepts) in enumerate(test_dl):
              labels = labels.to(torch.long).to(device)
              concepts = concepts.to(device)
              if pair == 'decoy':
                  concept1 = torch.rand(concepts.size(0), 1).to(device).squeeze(1)
                  concept2 = torch.rand(concepts.size(0), 1).to(device).squeeze(1)
                  preds = model(concept1, concept2)
              else:
                preds = model(concepts[:,concept1_idx], concepts[:,concept2_idx])
              y_pred_list = preds.argmax(dim=1).tolist()

              for y in y_pred_list:
                 y_pred.append(y)
              y_true_list = labels.tolist()
              for y in y_true_list:
                  y_true.append(y)

              '''
              # Manually inspect the predictions
              if version == 'labelsupervision':
                 if pair == 'informative':
                     for i, lab in enumerate(labels.tolist()):
                        print(i,lab,concepts[i,concept1_idx].item(), concepts[i,concept2_idx].item())      
              else:
                continue
              '''
          scores = classification_report(y_true, y_pred, output_dict=True)
          macro_avg = scores['macro avg']['precision']
          results.append(f'Model {version} {beta} | Macro avg precision for pair {pair}: {macro_avg}')
          
              
print('---- TERMINATED ------')
print('\n\n\n')
for r in results:
    print(r)
