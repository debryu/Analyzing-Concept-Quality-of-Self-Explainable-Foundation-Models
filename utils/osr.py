import torch
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
def get_osr_params(model, dataset, args):
        # Check if the prototypes are already saved
        if os.path.exists(os.path.join('data/osr', 'prototypes.pk')) and os.path.exists(os.path.join('data/osr', 'thr_y.pk')) and os.path.exists(os.path.join('data/osr', 'thr_rec.pk')):
            prototypes = pickle.load(open(os.path.join('data/osr', 'prototypes.pk'), 'rb'))
            thr_y = pickle.load(open(os.path.join('data/osr', 'thr_y.pk'), 'rb'))
            thr_rec = pickle.load(open(os.path.join('data/osr', 'thr_rec.pk'), 'rb'))
        else:
            thr_rec, thr_y, prototypes = update_osr(model, dataset, args)
        return thr_rec, thr_y, prototypes

def update_osr(model, dataset, args):
        # to evaluate performances on disentanglement
        # Run all the samples from train
        train_loader, val_loader, test_loader = dataset.get_data_loaders()
        model = model.to(model.device)
        model.eval()
        
        RECON_LOSSES = []
        PREDS = []
        MUS = []
        # The model output is this
        # {'MUS': mu, 'LOGVARS': logvar, 'LATENTS': latents, 'RECS': recs, 'PREDS': preds}
        print('Updating OSR')
        for internal_iter, (imgs, labels, concepts) in enumerate(train_loader):
            imgs, labels, concepts = imgs.to(model.device), labels.to(model.device), concepts.to(model.device)
            out_dict = model(imgs)
            
            # softmax the predictions
            preds = torch.nn.functional.softmax(out_dict['PREDS'], dim=1)
            # get the prediction
            preds = torch.argmax(preds, dim=1)
            mask = (preds == labels)
            mus = out_dict['MUS'][mask]
            recons = out_dict['RECS'][mask]
            imgs = imgs[mask]
            
            #MUS.append(out_dict['MUS'][mask])
            #RECON_IMGS.append(out_dict['RECS'][mask])
            #PREDS.append(preds[mask])
            #IMGS.append(imgs[mask])

            # Compute the batch loss
            recon_losses = torch.nn.functional.mse_loss(recons, imgs, reduction='none')
            # Compute the loss image-wise
            recon_losses = torch.mean(recon_losses, dim=(1,2,3))

            RECON_LOSSES.append(recon_losses.detach().cpu().numpy())
            MUS.append(mus.detach().cpu().numpy())
            PREDS.append(preds.detach().cpu().numpy())

        # Concatenate all the results
        RECON_LOSSES = torch.tensor(np.concatenate(RECON_LOSSES))
        PREDS = torch.tensor(np.concatenate(PREDS))
        MUS = torch.tensor(np.concatenate(MUS))

        classes = PREDS.unique().cpu().numpy().tolist()
        print('Classes:', classes)
            
        ## EVALUATE THE RECON THR ##    
        print('## Updating thr in rec')
        l = len(recon_losses)
        r_min, r_max = torch.min(recon_losses).item(), torch.max(recon_losses).item()
        good_r = []
        
        for eta in tqdm(np.linspace(r_min, r_max, 1000)):
            mask = (recon_losses < eta)
            if len( recon_losses[mask] )/l > 0.945 and len( recon_losses[mask] )/l < 0.955:
                good_r.append(eta)
        good_r = torch.tensor(good_r)
        thr_rec = torch.mean(good_r).to(model.device)
        print('Updated the threshold on reconstruction')
        
        
        ## EVALUATE THE zy THR ##
        print('## Updating thr in latent representations')
        good_dist = []
        prototypes = {}
        thr_y = {}
        for yclass in classes:
            mask = (preds == yclass)
            l = len(mus[mask])
            mean = torch.mean(mus[mask], dim=0)
            prototypes[yclass] = mean
            dist = [mean - k for k in mus[mask]]
            dist = [torch.linalg.vector_norm(k, ord=2).item() for k in dist]
            dist = np.array(dist)
            
            dmin = np.min(dist)
            dmax = np.max(dist)
            for eta in np.linspace(dmin, dmax, 10000):
                conds = dist < eta 
                # Count the number of True in conds
                if conds.sum()/l > 0.945 and  conds.sum()/l < 0.955:
                    good_dist.append(eta)
            thr_y[yclass] = np.mean(good_dist)

        print('Updated the threshold on latent representations')
        print('thr_rec:', thr_rec, 'thr_y:', thr_y)
        
        ## SAVE INFO TO FOLDER
        pickle.dump(thr_rec, open(os.path.join('data/osr', 'thr_rec.pk'), 'wb'))
        pickle.dump(thr_y, open(os.path.join('data/osr', 'thr_y.pk'), 'wb'))
        pickle.dump(prototypes, open(os.path.join('data/osr', 'prototypes.pk'), 'wb'))

        return thr_rec, thr_y, prototypes


def run_osr(recons, imgs, mus, thr_r, thr_y, prototypes, plot=False):
    recon_losses = torch.nn.functional.mse_loss(recons, imgs, reduction='none')
    # Compute the loss image-wise
    recon_losses = torch.mean(recon_losses, dim=(1,2,3))
    mask_r = (recon_losses < thr_r)
    
    classes = list(prototypes.keys())
    masks = []
    for c in classes:
        dist = [prototypes[c] - k for k in mus]
        dist = [torch.linalg.vector_norm(k, ord=2).item() for k in dist]
        m = (dist < thr_y[c])
        masks.append(m)
    # If a sample is inside at least one prototype, it is considered as a good sample
    mask_y = np.logical_or.reduce(masks)
    
    # The final mask is the the AND of the two masks, meaning that the sample is good if it is good in both the recon and the latent space
    mask = np.logical_and(mask_r.cpu().numpy(), mask_y)
    #print('Skipped ', 64-mask.sum().item(), ' images. (', 64-mask_y.sum().item(), ' images due to distribution threshold,', 64-mask_r.sum().item(), ' images due to reconstruction threshold.')    
    not_mask = np.logical_not(mask)
    # plot the images that are not skipped (showing only the good samples if there are more skipped than good samples)
    if plot:
        if mask.sum().item() < not_mask.sum().item():
            recons = recons[mask]
            text = 'Good samples'
            imgs = imgs[mask]
        else:
            recons = recons[not_mask]
            text = 'Skipped samples'
            imgs = imgs[not_mask]

        recons = recons.cpu().detach().numpy()
        imgs = imgs.cpu().detach().numpy()
        recons = np.moveaxis(recons, 1, -1)
        imgs = np.moveaxis(imgs, 1, -1)
        n = len(recons)
        recons = recons[:n]
        imgs = imgs[:n]
        if n < 2:
            print('Not enough images to plot')
        else:
            fig, ax = plt.subplots(n, 2, figsize=(10, 10))
            for i in range(n):
                ax[i, 0].imshow(recons[i])
                ax[i, 1].imshow(imgs[i])
            col1,col2 = ax[0]
            col1.set_title('Reconstructed samples')
            col2.set_title('Original samples')
            plt.suptitle(text)
            plt.tight_layout()     
            plt.axis('off')       
            plt.show()
    return mask
        
        