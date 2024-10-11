import torch
import os
import random
#import utils
import utils.data_utils as data_utils
import metrics.similarity as similarity
import argparse
import datetime
import json
import LF_CBM.utils as utils
import pickle
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset
from train_labo import train_LaBo
from utils.preprocessing_model_data import get_removed_indices

parser = argparse.ArgumentParser(description='Settings for creating CBM')

parser.add_argument("--model", type=str, default="lfcbm", help="Which model to train")
parser.add_argument("--dataset", type=str, default="shapes3d")
parser.add_argument("--concept_set", type=str, default=None, 
                    help="path to concept set name")
parser.add_argument("--backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
parser.add_argument("--clip_name", type=str, default="ViT-B/16", help="Which CLIP model to use")

parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
parser.add_argument("--batch_size", type=int, default=258, help="Batch size used when saving model/CLIP activations")
parser.add_argument("--saga_batch_size", type=int, default=256, help="Batch size used when fitting final layer")
parser.add_argument("--proj_batch_size", type=int, default=50000, help="Batch size to use when learning projection layer")

parser.add_argument("--feature_layer", type=str, default='layer4', 
                    help="Which layer to collect activations from. Should be the name of second to last layer in the model")
parser.add_argument("--activation_dir", type=str, default='LF_CBM/saved_activations', help="save location for backbone and CLIP activations")
parser.add_argument("--save_dir", type=str, default='data/ckpt_lf', help="where to save trained models")
parser.add_argument("--clip_cutoff", type=float, default=0.25, help="concepts with smaller top5 clip activation will be deleted")
parser.add_argument("--proj_steps", type=int, default=1000, help="how many steps to train the projection layer for")
parser.add_argument("--interpretability_cutoff", type=float, default=0.45, help="concepts with smaller similarity to target concept will be deleted")
parser.add_argument("--lam", type=float, default=0.0007, help="Sparsity regularization parameter, higher->more sparse")
parser.add_argument("--n_iters", type=int, default=1000, help="How many iterations to run the final layer solver for")
parser.add_argument("--print", action='store_true', help="Print all concepts being deleted in this stage")


'''
S3D activations
 tensor([0.2675, 0.3039, 0.2916, 0.2967, 0.3235, 0.3210, 0.3167, 0.2654, 0.2981,
        0.2875, 0.3159, 0.2926, 0.2929, 0.3127, 0.3014, 0.3113, 0.3173, 0.2848,
        0.2886, 0.2988, 0.2724, 0.2966, 0.2972, 0.2909, 0.2766, 0.3125, 0.3058,
        0.3279, 0.3056, 0.3041, 0.3175, 0.3137, 0.3137, 0.3212, 0.2945, 0.2999,
        0.3068, 0.3161, 0.3080, 0.3191, 0.3218, 0.2823, 0.2923, 0.3089, 0.2905,
        0.2753, 0.2952, 0.2982, 0.2457, 0.2652, 0.2848, 0.2803, 0.2555, 0.2728,
        0.2768, 0.2849, 0.2797, 0.2858, 0.2930, 0.2756, 0.2927, 0.3023, 0.2773,
        0.2627, 0.2771, 0.2855, 0.2668, 0.2409, 0.2784, 0.2881, 0.2867, 0.2839,
        0.2806, 0.2866, 0.2728, 0.2943, 0.3025, 0.2713, 0.2609, 0.2748, 0.3027,
        0.2781, 0.2829, 0.2930, 0.2681, 0.2642, 0.2676, 0.2565])
S3D interpretability
tensor([0.9134, 0.9108, 0.8538, 0.8344, 0.9083, 0.8619, 0.8853, 0.9032, 0.9077,
        0.8668, 0.8838, 0.9411, 0.9246, 0.9029, 0.9063, 0.8773, 0.8894, 0.9497,
        0.9057, 0.8730, 0.8923, 0.9111, 0.9333, 0.9244, 0.8688, 0.9166, 0.9364,
        0.9203, 0.8626, 0.8997, 0.8498, 0.8825, 0.9442, 0.9133, 0.9044, 0.8851,
        0.9129, 0.8222, 0.8564, 0.9383, 0.8742, 0.8894, 0.9376, 0.8884, 0.8963,
        0.8919, 0.9181, 0.9461, 0.9370, 0.9074, 0.8908, 0.9033, 0.8860, 0.8730,
        0.8701, 0.9516, 0.8856, 0.9148, 0.9466, 0.9182, 0.9219],

CelebA activations
tensor([0.2945, 0.3251, 0.3237, 0.3179, 0.3133, 0.2996, 0.3064, 0.2958, 0.3135,
        0.3120, 0.2927, 0.2886, 0.2949, 0.3001, 0.2927, 0.2906, 0.2864, 0.3043,
        0.2954, 0.2937, 0.2871, 0.2985, 0.2851, 0.3023, 0.2963, 0.2848, 0.3043,
        0.2887, 0.3111, 0.3008, 0.3077, 0.3022, 0.3094, 0.3050, 0.3077, 0.3005,
        0.3091, 0.3033, 0.2897, 0.2949, 0.2853, 0.2884, 0.2930, 0.2945, 0.2994,
        0.2820, 0.2826, 0.2972, 0.2989, 0.3090, 0.2921, 0.3165, 0.3129, 0.3104,
        0.2858, 0.3080, 0.3103, 0.2967, 0.3115, 0.3144, 0.2924, 0.2916, 0.3062,
        0.3166, 0.3096, 0.3172, 0.3180, 0.3193, 0.3013, 0.2988, 0.3086, 0.2957,
        0.3320, 0.3015, 0.2999, 0.2864, 0.3078, 0.3016, 0.3092, 0.2904, 0.2946,
        0.2997, 0.3098, 0.3158, 0.2905, 0.3079, 0.2889, 0.2853, 0.2929, 0.2983,
        0.3078, 0.3043, 0.2869, 0.2969, 0.3166, 0.2986, 0.3049, 0.2904, 0.2811])
CelebA interpretability
tensor([0.6040, 0.6716, 0.6923, 0.6208, 0.7152, 0.6386, 0.5970, 0.6446, 0.6751,
        0.6586, 0.6007, 0.6004, 0.7634, 0.7605, 0.4711, 0.6894, 0.5330, 0.7034,
        0.6657, 0.8394, 0.6795, 0.5998, 0.5442, 0.6167, 0.5083, 0.5609, 0.5694,
        0.5582, 0.6084, 0.5863, 0.5473, 0.5097, 0.5558, 0.5298, 0.5985, 0.5224,
        0.6401, 0.5877, 0.5771, 0.5609, 0.5621, 0.5795, 0.6069, 0.6103, 0.6592,
        0.5247, 0.4682, 0.5442, 0.5914, 0.7887, 0.6398, 0.7103, 0.6175, 0.6491,
        0.5143, 0.5732, 0.6014, 0.5218, 0.5924, 0.6715, 0.5509, 0.5715, 0.6585,
        0.7311, 0.6928, 0.7753, 0.6204, 0.7344, 0.6114, 0.8424, 0.7850, 0.5793,
        0.5158, 0.5952, 0.6338, 0.6083, 0.6282, 0.5517, 0.6065, 0.7747, 0.6446,
        0.7305, 0.8235, 0.8324, 0.5687, 0.7322, 0.6995, 0.5580, 0.5379, 0.5812,
        0.5765, 0.7038, 0.6941, 0.6810, 0.7360, 0.7240, 0.7209, 0.4666, 0.6547],
       device='cuda:0')        
'''
def train_cbm_and_save(args):
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if args.concept_set==None:
        args.concept_set = "LF_CBM/concept_sets/{}_handmade_concepts.txt".format(args.dataset)
    
    
    # Create the model's save directory
    save_name = "{}/{}_cbm_{}".format(args.save_dir, args.dataset, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    os.mkdir(save_name)
    # Set the activation directory inside the save directory
    args.activation_dir = os.path.join(save_name, 'activations')
    os.makedirs(os.path.join(save_name, 'LaBo'), exist_ok=True)

    similarity_fn = similarity.cos_similarity_cubed_single
    
    d_train = args.dataset + "_train"
    d_val = args.dataset + "_val"
    
    #get concept set
    '''
    cls_file = data_utils.LABEL_FILES[args.dataset]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
    
    with open(args.concept_set) as f:
        concepts = f.read().split("\n")
    ''' #TEMPORARY

    with open(args.concept_set) as f:
        concepts = f.read().split("\n")

    concepts_dict = {'raw_concepts': concepts, 'raw_dim': len(concepts)}

    classes = ['red pill', ' ']
    #save activations and get save_paths
    for d_probe in [d_train, d_val]:
        utils.save_activations(clip_name = args.clip_name, target_name = args.backbone, 
                               target_layers = [args.feature_layer], d_probe = d_probe,
                               concept_set = args.concept_set, batch_size = args.batch_size, 
                               device = args.device, pool_mode = "avg", save_dir = args.activation_dir)
        
    target_save_name, clip_save_name, text_save_name = utils.get_save_names(args.clip_name, args.backbone, 
                                            args.feature_layer,d_train, args.concept_set, "avg", args.activation_dir)
    val_target_save_name, val_clip_save_name, text_save_name =  utils.get_save_names(args.clip_name, args.backbone,
                                            args.feature_layer, d_val, args.concept_set, "avg", args.activation_dir)
    
    #load features
    with torch.no_grad():
        target_features = torch.load(target_save_name, map_location="cpu").float()
        
        val_target_features = torch.load(val_target_save_name, map_location="cpu").float()

        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        val_image_features = torch.load(val_clip_save_name, map_location="cpu").float()
        val_image_features /= torch.norm(val_image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()
        text_features /= torch.norm(text_features, dim=1, keepdim=True)
        
        clip_features = image_features @ text_features.T            # Namely, P
        val_clip_features = val_image_features @ text_features.T    # Namely, P_val

        del image_features, text_features, val_image_features
    
    # Language in a Bottle (LaBo) concepts are
    # - clip_features
    # - val_clip_features
    # Save them to disk
    torch.save(clip_features, os.path.join(save_name, 'LaBo/train_concepts.labo'))
    torch.save(val_clip_features, os.path.join(save_name, 'LaBo/val_concepts.labo'))
    
    #filter concepts not activating highly
    # For every concept (vary the samples, aka dim 0) take the top 5 values (the 5 samples that activated that concept the most)
    # Return the values and compute the mean to get the mean of the top 5 activations for each concept
    highest = torch.mean(torch.topk(clip_features, dim=0, k=5)[0], dim=0)
    print('Highest:', highest)
    
    #print('Stage 0', len(concepts))
    removed_concepts_id = []
    for i, concept in enumerate(concepts):
        if highest[i]<=args.clip_cutoff:
            print("Deleting {}, CLIP top5:{:.3f}".format(concept, highest[i]))
            removed_concepts_id.append(i)
    print(f'Removed concepts: {removed_concepts_id}')
    concepts = [concepts[i] for i in range(len(concepts)) if highest[i]>args.clip_cutoff]
    
    #save memory by recalculating
    del clip_features
    with torch.no_grad():
        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()[highest>args.clip_cutoff]
        text_features /= torch.norm(text_features, dim=1, keepdim=True)
    
        clip_features = image_features @ text_features.T                # This is the actual P, removing the concepts that are not activating highly
        del image_features, text_features
    
    val_clip_features = val_clip_features[:, highest>args.clip_cutoff]  # This is the actual P_val, removing the concepts that are not activating highly
    
    #learn projection layer
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts),
                                 bias=False).to(args.device)
    opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)
    
    indices = [ind for ind in range(len(target_features))]
    
    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    proj_batch_size = min(args.proj_batch_size, len(target_features))
    for i in range(args.proj_steps):
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outs = proj_layer(target_features[batch].to(args.device).detach())
        loss = -similarity_fn(clip_features[batch].to(args.device).detach(), outs)
        
        loss = torch.mean(loss)
        loss.backward()
        opt.step()
        if i%50==0 or i==args.proj_steps-1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features.to(args.device).detach())
                val_loss = -similarity_fn(val_clip_features.to(args.device).detach(), val_output)
                val_loss = torch.mean(val_loss)
            if i==0:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
                print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(best_step, -loss.cpu(),
                                                                                               -best_val_loss.cpu()))
                
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
            else: #stop if val loss starts increasing
                break
        opt.zero_grad()
        #if i==10:       #TEMPORARY TO MAKE IT FASTER
        #    break
    proj_layer.load_state_dict({"weight":best_weights})
    print("Best step:{}, Avg val similarity:{:.4f}".format(best_step, -best_val_loss.cpu()))
    
    #delete concepts that are not interpretable
    with torch.no_grad():
        outs = proj_layer(val_target_features.to(args.device).detach())
        sim = similarity_fn(val_clip_features.to(args.device).detach(), outs)
        print(sim)
        interpretable = sim > args.interpretability_cutoff
        
    #print('Stage 1', len(concepts))
    similarities = []
    print('Starting with:', len(concepts)) 
    removed_ids_second_round = []
    for i, concept in enumerate(concepts):
        similarities.append(sim[i].item())
        if sim[i]<=args.interpretability_cutoff:
            print("Deleting {}, Iterpretability:{:.3f}".format(concept, sim[i]))
            removed_ids_second_round.append(i)
    print('Ending with:',len(concepts))
    print(f'Removed concepts: {removed_ids_second_round} -> for interpretability')
    print('Size:', len(removed_ids_second_round))
    print(f'Similarities: {similarities}')
    concepts = [concepts[i] for i in range(len(concepts)) if interpretable[i]]
    concepts_dict['filtered_concepts'] = concepts

    rem_ids_2 = get_removed_indices(removed_ids_second_round, removed_concepts_id)
    rem_ids_final = rem_ids_2 + removed_concepts_id

    #concepts_dict['removed_concepts_id'] = removed_concepts_id
    concepts_dict['removed_concepts_id'] = rem_ids_final
    pickle.dump(rem_ids_final, open(os.path.join(save_name, 'removed_concepts_id.list'), 'wb'))
    pickle.dump(rem_ids_final, open(os.path.join(save_name, 'LaBo/removed_concepts_id.list'), 'wb'))
    pickle.dump(concepts_dict, open(os.path.join(save_name, 'concepts.dict'), 'wb'))
    del clip_features, val_clip_features
    #print(interpretable)
    #print(len(interpretable))
    W_c = proj_layer.weight[interpretable]
    print('Shape of the final layer',W_c.shape)
    print('Len of saved c_rem_ids:', len(rem_ids_final))
    #print('W_c:', W_c.shape)
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts), bias=False)
    proj_layer.load_state_dict({"weight":W_c})

    train_targets = data_utils.get_targets_only(d_train)
    
    val_targets = data_utils.get_targets_only(d_val)
   
    with torch.no_grad():
        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std
        
        train_y = torch.LongTensor(train_targets)

        indexed_train_ds = IndexedTensorDataset(train_c, train_y)

        val_c -= train_mean
        val_c /= train_std
        
        val_y = torch.LongTensor(val_targets)
        
        val_ds = TensorDataset(val_c,val_y)

    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)

    # Make linear model and zero initialize
    n_concepts_final_layer = train_c.shape[1]
    print(n_concepts_final_layer)
    # add n_concepts to the args
    setattr(args, 'n_concepts_final_layer', n_concepts_final_layer)
    linear = torch.nn.Linear(train_c.shape[1],len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    
    STEP_SIZE = 0.1
    ALPHA = 0.99
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = args.lam

    # Solve the GLM path
    output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.n_iters, ALPHA, epsilon=1, k=1,
                      val_loader=val_loader, do_zero=False, metadata=metadata, n_ex=len(target_features), n_classes = len(classes))
    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']
    
    
    torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    torch.save(W_c, os.path.join(save_name ,"W_c.pt"))
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))
    
    
    '''
    LaBo
    '''
    train_LaBo(save_name, args, train_targets, val_targets, len(classes))

    with open(os.path.join(save_name, "concepts.txt"), 'w') as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write('\n'+concept)
    
    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Save args as pickle
    pickle.dump(args, open(os.path.join(save_name, 'args.pkl'), 'wb'))
    
    with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
        out_dict = {}
        for key in ('lam', 'lr', 'alpha', 'time'):
            out_dict[key] = float(output_proj['path'][0][key])
        out_dict['metrics'] = output_proj['path'][0]['metrics']
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict['sparsity'] = {"Non-zero weights":nnz, "Total weights":total, "Percentage non-zero":nnz/total}
        json.dump(out_dict, f, indent=2)
    
if __name__=='__main__':
    args = parser.parse_args()
    train_cbm_and_save(args)