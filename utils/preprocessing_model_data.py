import torch
import numpy as np

INDEXES_TO_REMOVE = {
        'shapes3d':[20,21,22,23,24,25,26,27,28,29,38,39,40,41],   #Color, Shape
        'shapes3d_lfcbm': [13,14,15,16,17,30,31,32,33,34,37,38,39,40,42,58,59,60,61,62,73,74,75,76,77,80,81,82,83,84],#[17,21,34,42,61],   #Color, Shape
        'celeba': [0,1,2,3,4,6,7,9,12,16,18,19,21,23,27,28,29,32,33,35,36,37],
        # !!! Concepts 2-9-32 are removed not because of causal relation but because the dataset is probably biased !!! (Towawrds female)
        # !!! Concepts 3-7-12 are removed not because of causal relation but because the dataset is probably biased !!! (Towards male)
        # Other concepts are removed because they are relevant to the task:
                                        #0 5oclockshadow
                                        #1 archedeyebrows
                                        #5 biglips
                                        #16 gloatee
                                        #18 heavy makeup
                                        #21 (would be 22 but 20 was removed) moustache
                                        #35 (would be 36) wearing lipstick
                                        #36 (would be 37) wearing necklace
        'celeba_lfcbm': [4,15,25,26,31,32,35,42,45,46,50,51,54,55,56,58,60,62,64,66,67,69,70,71,75,76,80,81,85],
        }

''''
0 5_o_Clock_Shadow 1 Arched_Eyebrows 2 Attractive 3 Bags_Under_Eyes 4 Bald 5 Bangs 6 Big_Lips 7 Big_Nose 8 Black_Hair 9 Blond_Hair 10 Blurry 
11 Brown_Hair 12 Bushy_Eyebrows 13 Chubby 14 Double_Chin 15 Eyeglasses 16 Goatee 17 Gray_Hair 
18 Heavy_Makeup 19 High_Cheekbones 20 Male  21 Mouth_Slightly_Open 22 Mustache 23 Narrow_Eyes 24 No_Beard 25 Oval_Face 
26 Pale_Skin 27 Pointy_Nose 28 Receding_Hairline 29 Rosy_Cheeks 30 Sideburns 31 Smiling 32 Straight_Hair 
33 Wavy_Hair 
34 Wearing_Earrings 
35 Wearing_Hat 
36 Wearing_Lipstick 
37 Wearing_Necklace 
38 Wearing_Necktie 
39 Young
'''

def get_removed_indices(ind_to_remove:list, prev_rem_indexes_id:list, total_concepts = 99):
    '''
    We have an original list where we removed some indexes (contained in prev_rem_indexes_id)
    after the removal we obtain id_first_pass
    From id_first_pass we want to remove the indexes contained in ind_to_remove.
    We want to adjust the indexes in ind_to_remove so that they refer to the original list
    '''

    # Sort the indexes to remove for easier handling
    ind_to_remove.sort()
    prev_rem_indexes_id.sort()
    new_ids = []
    remaining = []
    for i in range(total_concepts):
      if i in prev_rem_indexes_id:
        continue
      else:
        remaining.append(i)

    for id in ind_to_remove:
      new_ids.append(remaining[id])
  
    return new_ids

def update_indices(indexes:list, removed_indices:list):
    '''
    Given a list, that had some indexed removed by "removed_indices", we want to get some new indexes "indexes" that also is referred to the original list
    Returns "updated_indices" that is the list of indexes referred to the new list after the removal of "removed_indices"
    '''
    # Sort the removed indices for easier handling
    removed_indices.sort()
    
    updated_indices = []
    for sample in indexes:
        # Skip the sample if it is in the list of removed indices
        if sum(1 for removed in removed_indices if removed == sample) > 0:
            continue

        # Calculate the number of removed indices that are less than the current sample index
        removed_count = sum(1 for removed in removed_indices if removed < sample)
        
        # Update the sample index by subtracting the count of removed indices before it
        print('adding',sample)
        updated_index = sample - removed_count
        updated_indices.append(updated_index)
    
    return updated_indices

def leakage_indices(original_count:int, removed:list):
    '''
    Given a list, and given a set of indices that were removed, we want to get a map of the original indices to the new indices

    Parameters:
    original_count: The number of elements in the original list
    final_count: The number of elements in the final list
    '''
    id_map = {}
    inverse_id_map = {}
    k = 0
    for i in range(original_count):
        if i not in removed:
            id_map[i] = k
            inverse_id_map[k] = i
            k += 1
    return id_map, inverse_id_map

def dci_indices(original_count:int, removed:list):
    '''
    Given a list, and given a set of indices that were removed, we want to get a map of the original indices to the new indices

    Parameters:
    original_count: The number of elements in the original list
    final_count: The number of elements in the final list
    '''
    id_map = {}
    inverse_id_map = {}
    k = 0
    for i in range(original_count):
        if i not in removed:
            id_map[i] = k
            inverse_id_map[k] = i
            k += 1
    return id_map, inverse_id_map

def update_dictionary(out_dict, dictionary_to_update, losses):
  '''
    Create the dataset to estimate the COMPLETENESS and LEAKAGE
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