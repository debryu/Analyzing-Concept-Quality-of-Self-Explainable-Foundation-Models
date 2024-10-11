import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pickle
import numpy as np  

dir = 'data/stats'

data = {}
#for all files in the directory
for file in os.listdir(dir):
  file_dict = {}
  seed = file.split('_')[3].split('d')[1]
  beta = file.split('_')[2].split('a')[1]
  
  file_dict['seed'] = seed
  file_dict['file'] = pickle.load(open(f'{dir}/{file}', 'rb'))
  model_name = file.split('_')[1]
  if model_name == 'betaglancenet':
    model_name = 'GN-' + beta
  if model_name == 'cbmbase':
    model_name = 'CBM'
  if model_name not in data.keys():
    data[model_name] = [file_dict]
  else:
    data[model_name].append(file_dict)
  
colors = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']
patches = []
models = data.keys()
fig, ax = plt.subplots(2)
fig.set_size_inches(8, 5)

for i,model in enumerate(models):
  runs = data[model]
  first_iter = True
  for run in runs:
    stats = run['file']
    ce = stats['cross_entropy_concepts']
    dci = stats['DCI_disentanglement']
    accuracy = stats['concept_accuracy']
    
    color = colors[i]
    increasing_ce = [1-l for l in ce]
    #mask = np.where(np.array(increasing_ce) < 0.5)
    #increasing_ce = np.delete(increasing_ce, mask)
    #dci = np.delete(dci, mask)
    #accuracy = np.delete(accuracy, mask)
    # Shrink current axis's height by 10% on the bottom
    ax[0].scatter(increasing_ce, dci, label=f'{model}', c=color, s=6)
    ax[1].scatter(increasing_ce, accuracy, label=f'{model}', c=color, s=6)
    # Create space at the bottom of the plot for the legend
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0 + box.height * 0.03,
                    box.width, box.height * 0.97])
    # Set the y-axis name for the second plot to 'Concept Accuracy'
    ax[0].set_ylabel('DCI Disentanglement')
    ax[1].set_ylabel('Concept Accuracy')
    # Create the legend
    patch = mpatches.Patch(color=color, label=model)
    if first_iter:
      patches.append(patch)
      first_iter = False
    
    
    


plt.xlabel('CE-loss (1-ce)')
#plt.ylabel('DCI Disentanglement')
fig.legend(handles=patches,loc='upper left', bbox_to_anchor=(0.12, 0.07), ncol = len(models))
plt.show()
