import matplotlib.pyplot as plt
import os
import pickle

# Stats folder position
stats_folder = './data/stats'

def plot_losses(metrics, save_path, name, save_plot=True):
    '''
    Plot the losses of the model
    '''
    model = metrics['model']
    beta = metrics['beta']
    seed = metrics['seed']
    dataset = metrics['dataset']  
    old_naming_file = f"eval_losses_{model}_beta{beta}_seed{seed}.history"
    new_naming_file = f"eval_losses_{dataset}_{model}_beta{beta}_seed{seed}.history"
    #print(f"Old naming: {old_naming_file}")
    #print(f"New naming: {new_naming_file}")

    #print(files)
    old_naming_file = os.path.join(stats_folder, old_naming_file)
    new_naming_file = os.path.join(stats_folder, new_naming_file)
    if os.path.exists(old_naming_file):
        print('Old naming exists')
        losses = pickle.load(open(old_naming_file,'rb'))
    elif os.path.exists(new_naming_file):
        print('New naming exists')
        losses = pickle.load(open(new_naming_file,'rb'))
    else:
        print('No history has been found!')
        return
    
    #print(losses)
    
    n_epochs = len(losses)
    x_axis = list(range(n_epochs))
    recon = []
    recon_present = False
    kld = []
    kld_present = False
    c = []
    pred = []
    for i,epoch in enumerate(losses):
      #print(epoch)
      if 'testrecon-loss' in epoch:
        recon.append(epoch['testrecon-loss'])
        recon_present = True
      if 'testkld' in epoch:
        kld.append(epoch['testkld'])
        kld_present = True
      c.append(epoch['testc-loss'])
      pred.append(epoch['testpred-loss'])

    plt.figure(figsize=(10,5))
    if recon_present:
      plt.plot(x_axis,recon, label='Reconstruction loss')
    #plt.plot(x_axis,kld, label='KLD loss')
    plt.plot(x_axis,c, label='Concepts loss')
    plt.plot(x_axis,pred, label='Label prediction loss')
   
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if save_plot:
        plt.savefig(os.path.join(save_path,name))
    else:
        plt.show()
    
    return


