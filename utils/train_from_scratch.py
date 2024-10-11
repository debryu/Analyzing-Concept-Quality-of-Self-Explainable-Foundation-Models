import torch
import numpy as np
import wandb
import os
from tqdm import tqdm
from torchvision.utils import make_grid
from utils.wandb_logger import *
from utils.status import progress_bar
from datasets.utils.base_dataset import BaseDataset
from utils.generative import conditional_gen, recon_visaulization
from metrics.dci_framework import _compute_dci
from utils.checkpoint import CheckpointSaver
from utils.log_images import *
from metrics.evaluate_val_test import evaluate_metrics
from utils.utils import ramping_beta
import pickle
from warmup_scheduler import GradualWarmupScheduler
from utils.cyclic_annealing import Cyclic_Annealer
from torch import nn

VAE_MODELS = ['betaglancenet','betavae','conceptvae','cvae']


def train(model, dataset: BaseDataset, _loss, args, use_prior = False, prior_weight = 100):
    eval_losses_history = []
    # Log everything
    train_logs = {'concept_accuracy':[], 'cross_entropy_concepts':[],'DCI_disentanglement':[], 'other':[]}

    # saving location
    ckpt_saver = CheckpointSaver(args.checkout, decreasing=True, args=args)

    # Patience for early stopping
    patience = 0
    max_patience = 6
    
    # Default Setting for Training
    model.to(model.device)
    train_loader, val_loader, (test_loader, ood_loader) = dataset.get_data_loaders()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(model.opt, args.exp_decay)
    #w_scheduler = GradualWarmupScheduler(model.opt, 1.0, args.warmup_steps)

    if args.wandb is not None:
        print('\n---wandb on\n')
        wandb.init(project="thesis", entity=args.wandb, 
                   name=str(args.dataset)+"_"+str(args.model),
                   config=args)
        wandb.config.update(args)

    print('\n--- Start of Training ---\n')

    # default for warm-up
    model.opt.zero_grad()
    model.opt.step()

    CE_weight = None
    CE_weight_labels = None
    if os.path.exists(f'data/ds_freqs/{args.dataset}_freq.pkl'):
        ds_freq = pickle.load(open(f'data/ds_freqs/{args.dataset}_freq.pkl', 'rb'))
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
        pickle.dump(ds_freq, open(f'data/ds_freqs/{args.dataset}_freq.pkl', 'wb'))

        print('Concept CE weight:', CE_weight)
        print('Concept freq:', frequencies)
        print('CE_weight_labels:', CE_weight_labels)
    
    #ramping_beta_target = None
    #if args.beta < 0:
    #    ramping_beta_target = abs(args.beta)
    #    args.beta = 0
    #else:
    #    ramping_beta_target = args.beta
    #cyc_annealer = Cyclic_Annealer(total_iters= args.n_epochs * len(train_loader), n_cycles=4, beta=args.beta, ratio = 0.5, function=lambda x: x*2, z_capacity_base=args.z_capacity, z_capacity_step=0)
    cyc_annealer = Cyclic_Annealer(total_iters= args.n_epochs * len(train_loader), n_cycles=4, beta=args.beta, ratio = 0.5, function=lambda x: x*2, z_capacity_base=args.z_capacity*3, z_capacity_step=args.z_capacity)
    
    for epoch in range(args.n_epochs):
        model.train()
        train_losses = []
        for i, data in enumerate(train_loader):
            if args.dataset == 'celeba': # Should not be needed anymore
                #images, (labels, concepts) = data 
                images, labels, concepts = data #Should be fixed now
                images, labels, concepts = images.to(model.device), labels.to(model.device), concepts.to(model.device)

                # Data augmentation
                gaussian_noise = torch.randn_like(images) * 0.05
                images = images + gaussian_noise
                # Plot the images with matplotlib
                #plt.imshow(images[0].permute(1,2,0).cpu().numpy())
                #plt.show()
            elif args.dataset == 'shapes3d': # Should not be needed anymore
                #images, (labels, concepts) = data 
                images, labels, concepts = data #Should be fixed now
                images, labels, concepts = images.to(model.device), labels.to(model.device), concepts.to(model.device)
                
                # Data augmentation
                gaussian_noise = torch.randn_like(images) * 0.01
                images = images + gaussian_noise
                # Plot the images with matplotlib
                #plt.imshow(images[0].permute(1,2,0).cpu().numpy())
                #plt.show()

            else:
                images, labels, concepts = data 
                images, labels, concepts = images.to(model.device), labels.to(model.device), concepts.to(model.device)
            # Mask the concepts
            # Only keep scale and shape, not color (orientation already masked in the dataset)
            #{'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 'scale': 8, 'shape': 4, 'orientation': 15}

            if args.sup_version == '12sup':
                concepts[:, :30] = -1
            elif args.sup_version == '30sup':
                concepts[:, 30:] = -1
            
            # CODE --------------------| Description --------------------------------------------| Old naming system  | New naming system
            # No change                | No label supervision                                    |         -          | nolabel
            # No change                | Full supervision on all concepts                        | CASE 0             | fullsup
            #concepts[:, :30] = -1    #| # Supervision on 12 concepts                            | CASE 1             | 12sup
            #concepts[:, 30:] = -1    #| # Supervision on 30 concepts                            | CASE 2             | 30sup
            #concepts[:, 38:] = -1   # Supervision on everything but shape                      | CASE 2             |   -
            #concepts[:, :20] = -1  # Supervision on useful concepts                            | CASE 3             |   -
           
            #concepts[:, 20:30] = -1 
            #concepts[:, 38:] = -1   # Supervision on useless concepts                          | CASE 4             |   -

            #concepts[:, 20] = -1
            #concepts[:, 41] = -1   # Supervision on useless concepts (everything but red pills)| CASE 5             | badsup
            
            
            out_dict = model(images)
            
            out_dict.update({'mu_cluster': None})
            if use_prior and args.model in VAE_MODELS:
                mask = (concepts[0,:] != -1)
                prior = model.compute_prior(labels)
                print('Prior:', prior)
                out_dict.update({'mu_cluster': prior})
                predicted_mus = torch.tanh(out_dict['mu_cluster']/2)
                loss_bin = nn.BCELoss(reduction='mean')((1+predicted_mus[:, mask])/2, concepts.to(dtype = torch.float)[:, mask]) * prior_weight

            out_dict.update({'INPUTS': images, 'LABELS': labels, 'CONCEPTS': concepts, 'CE_WEIGHT': CE_weight, 'CE_weight_labels': CE_weight_labels})
            
            #if i == 0 and args.model in ['betaglancenet','betavae','conceptvae','cvae']: save_imgs(out_dict, args, epoch, max_images=2)
            
            if args.sup_version == '12sup' and args.model in VAE_MODELS:
                out_dict['MUS'], out_dict['LOGVARS'] = out_dict['MUS'][:, :30], out_dict['LOGVARS'][:, :30]
                if use_prior and args.model in VAE_MODELS:
                    out_dict['mu_cluster'] = out_dict['mu_cluster'][:, :30] 
                
            elif args.sup_version == '30sup' and args.model in VAE_MODELS:
                out_dict['MUS'], out_dict['LOGVARS'] = out_dict['MUS'][:, 30:], out_dict['LOGVARS'][:, 30:]
                if use_prior and args.model in VAE_MODELS:
                    out_dict['mu_cluster'] = out_dict['mu_cluster'][:, 30:]
                


            model.opt.zero_grad()
            loss, losses = _loss(out_dict, args)
            if use_prior and args.model in VAE_MODELS:
                if args.beta > 0:
                    loss = loss + loss_bin

            train_losses.append(loss.item())
            loss.backward()
            model.opt.step()
            if args.model in VAE_MODELS:
                args.beta, args.z_capacity = cyc_annealer.step()
            if i%1000 == 0:
                print('\nBETA:', args.beta, 'Z_CAPACITY:', args.z_capacity)

            if args.wandb is not None:
                wandb_log_step(epoch, loss.item(), args.beta, args.z_capacity, losses)
            
            if i % 10 ==0: progress_bar(i, len(train_loader)-9, epoch, np.mean(train_losses))
            #if i > 400: break # for debugging
        
        model.eval()
        tloss, cacc, yacc, _ = evaluate_metrics(model, val_loader, _loss, args, use_prior=use_prior)
        eval_losses_history.append(tloss)
        #train_logs['concept_accuracy'].append(cacc)
        #train_logs['cross_entropy_concepts'].append(tloss['testc-loss'])
        
        # update at end of the epoch 
        #if epoch < args.warmup_steps:  
        #    print('   LR:', w_scheduler.get_last_lr())
        #    if ramping_beta_target is not None:
        #        args = ramping_beta(epoch=epoch, args=args, target=ramping_beta_target, total_steps=2)
        #    w_scheduler.step()
        #else:          
        #    print('   LR:', scheduler.get_last_lr())
        #    scheduler.step()
        #    if ramping_beta_target is not None:
        #        args = ramping_beta(epoch=epoch-args.warmup_steps, args=args, target=ramping_beta_target, total_steps=args.n_epochs-10)
                #status = None
                #if(args.beta > 0.1):
                    # If ramping beta, save the model not considering the first iteration where beta is 0 
                    # This creates the problem that with beta = 0 the loss will be minimal
                    #status = ckpt_saver(model, epoch, tloss, args)
            #else: 
                # If no ramping beta, save the model without issues
                #status = ckpt_saver(model, epoch, tloss, args)
            #if status == 'failed' and epoch > 30:
                #patience += 1
                #if patience > max_patience:
                    #print('Early stopping')
                    #break
            #else:
                #patience = 0

        ### LOGGING ###
        print(' dict of losses: \n', tloss)
        print('  ACC C', cacc, '  ACC Y', yacc, ' BETA:', args.beta)

        if args.wandb is not None:
            wandb_log_epoch(epoch=epoch, acc=yacc, cacc=cacc,
                            tloss=tloss,
                            lr=float(scheduler.get_last_lr()[0]))
            
            if hasattr(model, 'decoder'):
                list_images = make_grid(conditional_gen(model,out_dict['mu_cluster'], args), nrow=8, )
                images = wandb.Image(list_images, caption="Generated samples")
                wandb.log({"Conditional Gen": images})

                list_images = make_grid(recon_visaulization(out_dict), nrow=8)
                images = wandb.Image(list_images, caption="Reconstructed samples")
                wandb.log({"Reconstruction": images})
    
    ## LAST SAVE
    ckpt_saver(model, epoch, tloss, args)    # Temporarily disabled
    
    # Temporary set this to every epoch
    #----------------------------------------------------------------------------------------------------
    '''
        # Evaluate performances on val or test
        if args.validate:
            c_true, c_pred, g_true = evaluate_metrics(model, test_loader, _loss, args, loc='val')   #what's the difference between c_true and g_true?
        else:
            loss, c_acc, _, (c_pred, g_true) = evaluate_metrics(model, test_loader, _loss, args, loc='test')
        
        #train_logs = {'concept_accuracy':[], 'cross_entropy_concepts':[],'DCI_disentanglement':[], 'other':[]}

        if not args.validate:
            #align, W = alignment_score(c_pred, c_true)
            #align = np.max([0, align])

            L = len(g_true)
            #Randomly shuffle the data to prevent having only one class in the training set
            #idx = np.random.permutation(L)
            #c_pred = c_pred[idx]
            #g_true = g_true[idx]

            z_train = c_pred[: round(L*0.7)]
            z_test  = c_pred[round(L*0.7) :]
            g_train = g_true[: round(L*0.7)]
            g_test  = g_true[round(L*0.7) :]
            # Temporary to make it faster
            z_train = c_pred[: 1000]
            z_test  = c_pred[1000 :2000]
            g_train = g_true[: 1000]
            g_test  = g_true[1000 :2000]

            if args.dataset == 'dsprites':
                g_train = g_train[:, 1:]
                g_test  = g_test[:, 1:]

            dci_metrics = _compute_dci(z_train.T, g_train.T, z_test.T, g_test.T)
            #train_logs = {'concept_accuracy':[], 'cross_entropy_concepts':[],'DCI_disentanglement':[], 'other':[]}
            train_logs['DCI_disentanglement'].append(dci_metrics['disentanglement'])
            train_logs['other'].append(dci_metrics.pop('disentanglement'))
            #print('Alignment:', align)
            #print('DCI SCORES: \n\n',  dci_metrics)

        else: 
            align, W = alignment_score(c_pred[:2000], c_true[:2000])
            align = np.max([0, align])

            z_train = c_pred[:1000]
            z_test  = c_pred[1000:]

            g_train = g_true[:1000]
            g_test  = g_true[1000:]
            # remove the first latent from computation
            if args.dataset == ['dsprites']:
                g_train = g_train[:, 1:]
                g_test  = g_test[:, 1:]

            dci_metrics = _compute_dci(z_train.T, g_train.T, z_test.T, g_test.T)
            
            print('Alignment:', align)
            print('DCI SCORES: \n\n',  dci_metrics)

    '''   
       
    #----------------------------------------------------------------------------------------------------
    # Save logs
    #if ramping_beta_target is not None:
    #    beta_str = str(int(ramping_beta_target)) + 'ramp'
    #else:
    beta_str = str(args.beta)
    pickle.dump(train_logs, open(f'data/stats/logs_{args.model}_beta{beta_str}_seed{args.seed}.pkl', 'wb'))
    pickle.dump(eval_losses_history, open(f'data/stats/eval_losses_{args.dataset}_{args.model}_beta{beta_str}_seed{args.seed}.history', 'wb'))

    
    if args.wandb is not None:
        '''
        wandb.log( {'align': align} )
        h_matrix = wandb.Image(W, caption='Hinton matrix')
        wandb.log({'hinton_matrix': h_matrix})

        wandb.log(dci_metrics)

        if hasattr(model, 'decoder'):
            list_images = make_grid(conditional_gen(model), nrow=8,)
            images = wandb.Image(list_images, caption="Generated samples")
            wandb.log({"Conditional Gen": images})

            list_images = make_grid( recon_visaulization(out_dict), nrow=8)
            images = wandb.Image(list_images, caption="Reconstructed samples")
            wandb.log({"Reconstruction": images})
        '''
        wandb.finish()

    print('--- Training Finished ---')

        
        