import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.hinton import hinton
import pickle
from utils.lang_short import VOCAB

def save_IM_as_img(save_path,name,title,importance_matrix,save_plot=True):
  dim1,dim2 = importance_matrix.shape
  visualise(save_path, name, importance_matrix, (dim1,dim2), title, save_plot=save_plot)
  #plt.savefig(save_path + "/importance_matrix.png")
  return 

def visualise(save_path, name, R, dims, title = 'plot', save_plot=False):
    # init plot (Hinton diag)
    x,y = dims
    fig, axs = plt.subplots(1, figsize=(x, y), facecolor='w', edgecolor='k')
    
    # visualise
    hinton(R, 'Z', 'C', ax=axs, fontsize=10)
    axs.set_title('{0}'.format(title), fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.rc('text', usetex=False)
    if save_plot:
        fig.tight_layout()
        plt.savefig(os.path.join(save_path,name))
    else:
        plt.show()


#importance_matrix = np.random.rand(10,10)

'''
TEST

'''

'''
# load importance matrix
path = "C:/Users/debryu/Desktop/VS_CODE/HOME/ML/Tirocinio/interpreter/data/ckpt/CBMbase_epoch49_seed0_beta1.0_2024-07-31_23-12-21"
save_path = os.path.join(path,"images")


analysis_path = os.path.join(path,"complete_analysis.dict")
analysis = pickle.load(open(analysis_path,'rb'))


dataset = analysis['dataset']
if dataset in VOCAB:
    dataset = VOCAB[dataset]

model = analysis['model']
if model in VOCAB:
    model = VOCAB[model]

version = analysis['version']
if version in VOCAB:
    version = VOCAB[version]

img_name = f"{model} {dataset} {version} importance matrix.png"
img_save_path = os.path.join(save_path,img_name)

IM = analysis['DCI']['importance_matrix']
save_IM_as_img(path,img_save_path, IM)
'''