import torch
import os
import pickle
import random

def train_LaBo(folder_path, args, train_labels, val_labels, n_classes):
  # Load the LaBo concepts
  LaBo_train_concepts = torch.load(os.path.join(folder_path, 'LaBo/train_concepts.labo'))
  LaBo_val_concepts = torch.load(os.path.join(folder_path, 'LaBo/val_concepts.labo'))
  concepts_metadata = pickle.load(open(os.path.join(folder_path, 'concepts.dict'), 'rb'))
  '''
    Concepts_metadata keys:
    - raw_concepts
    - removed_concepts_id
    - filtered_concepts
    - raw_dim
  '''
  total_concetps = concepts_metadata['raw_dim']
  print('Total concepts:', total_concetps)  
  removed_concepts_id = pickle.load(open(os.path.join(folder_path, 'removed_concepts_id.list'), 'rb'))#concepts_metadata['removed_concepts_id']
  print('Shape before concept filtering:')
  print(LaBo_train_concepts.shape, LaBo_val_concepts.shape)

  # Also train a linear model on the LaBo concepts
  # Mask filtered concepts
  mask = torch.ones(total_concetps, dtype=torch.bool)
  mask[removed_concepts_id] = False

  LaBo_train_concepts = LaBo_train_concepts[:,mask]
  LaBo_val_concepts = LaBo_val_concepts[:,mask]
  print('Shape after concept filtering:')
  print(LaBo_train_concepts.shape, LaBo_val_concepts.shape)

  train_labels = torch.tensor(train_labels, dtype=torch.long).to(args.device)
  val_labels = torch.tensor(val_labels, dtype=torch.long).to(args.device)

  #learn LaBo linear model
  LaBo_model = torch.nn.Linear(in_features=LaBo_train_concepts.shape[1], out_features=n_classes, bias=False).to(args.device)
  opt = torch.optim.Adam(LaBo_model.parameters(), lr=1e-3)
  
  indices = [ind for ind in range(len(LaBo_train_concepts))]
  
  best_val_loss = float("inf")
  best_step = 0
  best_weights = None
  proj_batch_size = args.proj_batch_size
  for i in range(args.proj_steps):
      batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
      outs = LaBo_model(LaBo_train_concepts[batch].to(args.device).detach())
      loss = torch.functional.F.cross_entropy(outs,train_labels[batch])
      
      loss = torch.mean(loss)
      loss.backward()
      opt.step()
      if i%50==0 or i==args.proj_steps-1:
          with torch.no_grad():
              val_output = LaBo_model(LaBo_val_concepts.to(args.device).detach())
              val_loss = torch.functional.F.cross_entropy(val_output,val_labels)
          if i==0:
              best_val_loss = val_loss
              best_step = i
              best_weights = LaBo_model.weight.clone()
              
          elif val_loss < best_val_loss:
              best_val_loss = val_loss
              best_step = i
              best_weights = LaBo_model.weight.clone()
          else: #stop if val loss starts increasing
              break
      opt.zero_grad()
      #if i==10:       #TEMPORARY TO MAKE IT FASTER
      #    break

  # Save the LaBo model
  torch.save(best_weights, os.path.join(folder_path, 'LaBo/labo_weights.pt'))
  