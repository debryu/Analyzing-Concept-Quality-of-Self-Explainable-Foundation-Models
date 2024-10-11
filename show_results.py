from datasets.utils.celeba_base import check_dataset
from datasets.celeba import CelebA
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import argparse
import pickle
import os
from collect_metrics import get_saved_models_to_run, run_analysis
from LF_CBM.collect_metrics import get_saved_models_to_run_lf, run_analysis_lf
from utils.save_importance_matrix import save_IM_as_img
from utils.plot_training_losses import plot_losses
from utils.lang_short import VOCAB
from utils.utils import leakage_function
import warnings

def output_images(model_metadata, overwrite = False):
    path = model_metadata['folder']
    images_path = os.path.join(path,'images') 
    if not os.path.exists(images_path) or overwrite:
        os.makedirs(images_path, exist_ok=True)
        dataset = model_metadata['dataset']
        if dataset in VOCAB:
            dataset = VOCAB[dataset]

        model = model_metadata['model']
        if model in VOCAB:
            model = VOCAB[model]

        version = model_metadata['version']
        if version in VOCAB:
            version = VOCAB[version]

        title = f"{model} {dataset} {version} importance matrix"
        title = 'test'
        IM = model_metadata['DCI']['importance_matrix']
        save_IM_as_img(images_path,'importance_matrix.png', title, IM, save_plot=True)

        #plot_losses(model_metadata, images_path, 'losses.png', save_plot=True)  
          
    

def load_metrics(model_metadata, exclude = {}):
    # Collect the args
    args = pickle.load(open(model_metadata['args'], 'rb'))
    # Check if the model should be excluded
    for key in exclude:
        if key in vars(args).keys():  #Convert the args to a dictionary to get the keys
            if getattr(args,key) in exclude[key]:
              print(f'Skipping model {args.model} because of {key} = {exclude[key]}')
              return None
        if key in model_metadata.keys():
            if model_metadata[key] in exclude[key]:
              print(f'Skipping model {args.model} because of {key} = {exclude[key]}')
              return None

    if not os.path.exists(os.path.join(model_metadata['folder'], 'complete_analysis.dict')):
        warnings.warn('No complete analysis for this model, running collect_metrics.py')
        metrics = run_analysis(model_metadata)
    else:
        metrics = pickle.load(open(os.path.join(model_metadata['folder'], 'complete_analysis.dict'), 'rb'))
    return metrics

def output_latex(metrics):
    seed = metrics['seed']
    model_type = metrics['model'] 
    version = metrics['version']
    beta = metrics['beta']
    DCI_dis_latex = metrics['DCI_disentanglement']
    CS_latex = metrics['CS']
    Leakage_CE_latex = metrics['CE_labels']
    Leakage_f1_latex = metrics['LKG_interpretable_score']
    Leakage_f1_linear_latex = metrics['LKG_interpretable_score_linear']

    CE_LKG_total = metrics['loss_Y_given_C']
    CE_LKG_uninformed = metrics['CE_labels']
    CE_LKG_uninformed_linear = metrics['CE_labels_linear']
    CE_LKG_random = metrics['CE_random_predictor']

    print(f'{CE_LKG_total} {CE_LKG_uninformed_linear} {CE_LKG_random}')
    print(f'{CE_LKG_total} {CE_LKG_uninformed} {CE_LKG_random}')
    
    lkg = leakage_function(CE_LKG_total, CE_LKG_uninformed, CE_LKG_random)
    lkg_linear = leakage_function(CE_LKG_total, CE_LKG_uninformed_linear, CE_LKG_random)

    CE_concepts_latex = metrics['CE_concepts']
    CE_labels_latex =  metrics['CE_labels']
    f1_score_labels_latex = metrics['label_report']['macro avg']['f1-score']
    f1_score_labels_post = metrics['best_report_Y_given_C']['macro avg']['f1-score']
    z_capacity = int(metrics['z_capacity'])
    w_c = metrics['w_c']
    dataset = metrics['dataset']
    # Handle abbreviations and the need to add beta to the model name
    if dataset in VOCAB:
        dataset = VOCAB[dataset]
    if model_type == 'betaglancenet':
        model_abbrev = f'BGN $\\beta{beta}$'
    elif model_type == 'cbmbase':
        model_abbrev = f'CBM'
    else:
        model_abbrev = model_type
    
    if z_capacity == 50 or z_capacity == '50':
        print(metrics)

    latex = f'{model_abbrev}.{dataset} & {version} & {z_capacity} & {w_c} & {DCI_dis_latex:.3f} & {CS_latex:.3f} & {Leakage_f1_linear_latex:.1f}-{lkg_linear:.2f} & {Leakage_f1_latex:.1f}-{lkg:.2f} & {CE_concepts_latex:.1e} & {f1_score_labels_latex:.3f}-{f1_score_labels_post:.3f} \\\\ \n'
    return latex,seed

def output_lfcbm_latex(metrics):
    seed = "-"
    model_type = metrics['model']
    version = metrics['version']
    beta = "-"
    DCI_dis_latex = metrics['DCI_disentanglement']
    CS_latex = metrics['CS']
    Leakage_CE_latex = metrics['CE_labels']
    Leakage_f1_latex = abs(2*(metrics['f1-score-leakage']/metrics['label_report']['macro avg']['f1-score'])-1)
    #f1-score-leakage_linear
    #
    #Leakage_f1_latex = metrics['LKG_interpretable_score']
    Leakage_f1_linear_latex = abs(2*(metrics['f1-score-leakage_linear']/metrics['label_report']['macro avg']['f1-score'])-1)
    #
    if Leakage_f1_latex > 1:
        Leakage_f1_latex = 1
    if Leakage_f1_linear_latex > 1:
        Leakage_f1_linear_latex = 1

    CE_LKG_total = metrics['loss_Y_given_C']
    CE_LKG_uninformed = metrics['CE_labels']
    CE_LKG_uninformed_linear = metrics['CE_labels_linear']
    CE_LKG_random = metrics['CE_random_predictor']

    print(f'{CE_LKG_total} {CE_LKG_uninformed_linear} {CE_LKG_random}')
    print(f'{CE_LKG_total} {CE_LKG_uninformed} {CE_LKG_random}')
    
    lkg = leakage_function(CE_LKG_total, CE_LKG_uninformed, CE_LKG_random)
    lkg_linear = leakage_function(CE_LKG_total, CE_LKG_uninformed_linear, CE_LKG_random)
    # Set this to NaN
    CE_concepts_latex = float('nan')
    CE_labels_latex =  metrics['CE_labels']
    f1_score_labels_latex = metrics['label_report']['macro avg']['f1-score']
    # Handle abbreviations and the need to add beta to the model name
    if model_type == 'betaglancenet':
        model_abbrev = f'BGN $\\beta{beta}$'
    elif model_type == 'cbmbase':
        model_abbrev = f'CBM'
    else:
        model_abbrev = model_type
    
    latex = f'{model_abbrev} & {version} & - & - &{DCI_dis_latex:.3f} & {CS_latex:.3f} & {Leakage_f1_linear_latex:.1f}-{lkg_linear:.2f} & {Leakage_f1_latex:.1f}-{lkg:.2f} & {CE_concepts_latex:.1e} & {f1_score_labels_latex:.3f} \\\\ \n'
    return latex,seed

def output_labo_latex(metrics):
    seed = "-"
    model_type = metrics['model']
    version = metrics['version']
    beta = "-"
    DCI_dis_latex = metrics['DCI_disentanglement']
    CS_latex = metrics['CS']
    Leakage_CE_latex = metrics['CE_labels']
    Leakage_f1_latex = abs(2*(metrics['f1-score-leakage']/metrics['label_report']['macro avg']['f1-score'])-1)
    #f1-score-leakage_linear
    #
    #Leakage_f1_latex = metrics['LKG_interpretable_score']
    Leakage_f1_linear_latex = abs(2*(metrics['f1-score-leakage_linear']/metrics['label_report']['macro avg']['f1-score'])-1)
    #
    if Leakage_f1_latex > 1:
        Leakage_f1_latex = 1
    if Leakage_f1_linear_latex > 1:
        Leakage_f1_linear_latex = 1
    # Set this to NaN

    CE_LKG_total = metrics['loss_Y_given_C']
    CE_LKG_uninformed = metrics['CE_labels']
    CE_LKG_uninformed_linear = metrics['CE_labels_linear']
    CE_LKG_random = metrics['CE_random_predictor']

    print(f'{CE_LKG_total} {CE_LKG_uninformed_linear} {CE_LKG_random}')
    print(f'{CE_LKG_total} {CE_LKG_uninformed} {CE_LKG_random}')
    
    lkg = leakage_function(CE_LKG_total, CE_LKG_uninformed, CE_LKG_random)
    lkg_linear = leakage_function(CE_LKG_total, CE_LKG_uninformed_linear, CE_LKG_random)

    CE_concepts_latex = float('nan')
    CE_labels_latex =  metrics['CE_labels']
    f1_score_labels_latex = metrics['label_report']['macro avg']['f1-score']
    # Handle abbreviations and the need to add beta to the model name
    if model_type == 'betaglancenet':
        model_abbrev = f'BGN $\\beta{beta}$'
    elif model_type == 'cbmbase':
        model_abbrev = f'CBM'
    else:
        model_abbrev = model_type
    
    latex = f'{model_abbrev} & {version} & - & - &{DCI_dis_latex:.3f} & {CS_latex:.3f} & {Leakage_f1_linear_latex:.1f}- & {Leakage_f1_latex:.1f} & {CE_concepts_latex:.1e} & {f1_score_labels_latex:.3f} \\\\ \n'
    return latex,seed

def print_results_latex(metrics,text):
    if metrics['model'] == 'lfcbm':
        line,seed = output_lfcbm_latex(metrics)
    else:
        line,seed = output_latex(metrics)   
    text += line
    text += '\\hline \n'
    return text

def print_labo_results_latex(metrics,text):
    line, seed = output_labo_latex(metrics)
    text += line
    text += '\\hline \n'
    return text


print('Starting the analysis')
parser = argparse.ArgumentParser()
parser.add_argument("--reset", action='store_true', help="Overwrite the existing results")
parser.add_argument("--reset_lf", action='store_true', help="Overwrite the existing results")
args = parser.parse_args()
print('Parsed args')
# Current path
path = os.path.dirname(os.path.realpath(__file__))
models_metadata = get_saved_models_to_run(os.path.join(path,'data','ckpt'))
lf_models_metadata = get_saved_models_to_run_lf(os.path.join(path,'data','ckpt_lf'))
print('Loaded models metadata')
if args.reset or args.reset_lf:
    if args.reset:
        print('Resetting all the results!')
        for model_name in models_metadata:
            if os.path.exists(os.path.join(models_metadata[model_name]['folder'], 'complete_analysis.dict')):
                os.remove(os.path.join(models_metadata[model_name]['folder'], 'complete_analysis.dict'))
            if os.path.exists(os.path.join(models_metadata[model_name]['folder'], 'CS_dataset.ds')):
                os.remove(os.path.join(models_metadata[model_name]['folder'], 'CS_dataset.ds'))
            if os.path.exists(os.path.join(models_metadata[model_name]['folder'], 'LKG_dataset.ds')):
                os.remove(os.path.join(models_metadata[model_name]['folder'], 'LKG_dataset.ds'))
            if os.path.exists(os.path.join(models_metadata[model_name]['folder'], 'CS_results.dict')):
                os.remove(os.path.join(models_metadata[model_name]['folder'], 'CS_results.dict'))
            if os.path.exists(os.path.join(models_metadata[model_name]['folder'], 'LKG_results.dict')):
                os.remove(os.path.join(models_metadata[model_name]['folder'], 'LKG_results.dict'))
            if os.path.exists(os.path.join(models_metadata[model_name]['folder'], 'leakage_predictor.statedict')):
                os.remove(os.path.join(models_metadata[model_name]['folder'], 'leakage_predictor.statedict'))
            if os.path.exists(os.path.join(models_metadata[model_name]['folder'], 'leakage_linear_predictor.statedict')):
                os.remove(os.path.join(models_metadata[model_name]['folder'], 'leakage_linear_predictor.statedict'))
            if os.path.exists(os.path.join(models_metadata[model_name]['folder'], 'P_Y_given_C.statedict')):
                os.remove(os.path.join(models_metadata[model_name]['folder'], 'P_Y_given_C.statedict'))
            if os.path.exists(os.path.join(models_metadata[model_name]['folder'], 'P_Y_given_E.statedict')):
                os.remove(os.path.join(models_metadata[model_name]['folder'], 'P_Y_given_E.statedict'))
            if os.path.exists(os.path.join(models_metadata[model_name]['folder'], 'metrics.dict')):
                os.remove(os.path.join(models_metadata[model_name]['folder'], 'metrics.dict'))
    
    
    for model_name in lf_models_metadata:
        print('Reset LF-CBM results')
        if os.path.exists(os.path.join(lf_models_metadata[model_name]['folder'], 'complete_analysis.dict')):
            os.remove(os.path.join(lf_models_metadata[model_name]['folder'], 'complete_analysis.dict'))
        if os.path.exists(os.path.join(lf_models_metadata[model_name]['folder'], 'CS_dataset.ds')):
            os.remove(os.path.join(lf_models_metadata[model_name]['folder'], 'CS_dataset.ds'))
        if os.path.exists(os.path.join(lf_models_metadata[model_name]['folder'], 'LKG_dataset.ds')):
            os.remove(os.path.join(lf_models_metadata[model_name]['folder'], 'LKG_dataset.ds'))
        if os.path.exists(os.path.join(lf_models_metadata[model_name]['folder'], 'CS_results.dict')):
            os.remove(os.path.join(lf_models_metadata[model_name]['folder'], 'CS_results.dict'))
        if os.path.exists(os.path.join(lf_models_metadata[model_name]['folder'], 'LKG_results.dict')):
            os.remove(os.path.join(lf_models_metadata[model_name]['folder'], 'LKG_results.dict'))
        if os.path.exists(os.path.join(lf_models_metadata[model_name]['folder'], 'leakage_predictor.statedict')):
            os.remove(os.path.join(lf_models_metadata[model_name]['folder'], 'leakage_predictor.statedict'))
        if os.path.exists(os.path.join(lf_models_metadata[model_name]['folder'], 'P_Y_given_C.statedict')):
            os.remove(os.path.join(lf_models_metadata[model_name]['folder'], 'P_Y_given_C.statedict'))
        if os.path.exists(os.path.join(lf_models_metadata[model_name]['folder'], 'P_Y_given_E.statedict')):
            os.remove(os.path.join(lf_models_metadata[model_name]['folder'], 'P_Y_given_E.statedict'))
        if os.path.exists(os.path.join(lf_models_metadata[model_name]['folder'], 'metrics.dict')):
            os.remove(os.path.join(lf_models_metadata[model_name]['folder'], 'metrics.dict'))
        if os.path.exists(os.path.join(lf_models_metadata[model_name]['folder'], 'final_concepts.txt')):
            os.remove(os.path.join(lf_models_metadata[model_name]['folder'], 'final_concepts.txt'))
    

    
    raise Exception('Results have been reset, please run again without --reset or --reset_lf')  
      

all_metrics_LF = []
all_metrics = []
# Start the analysis
header = 'Model & Superv. & Capac. & W\\_C & DCI (dis) $\\uparrow$ & CS $\\uparrow$ & LKG (lin)$\\uparrow$ & LKG (f1) $\\downarrow$& CE Concepts $\\downarrow$& f1-score $\\uparrow$ \\\\'
latex_table = '\n\n' + header + '\n\\hline\n'

for model_name in lf_models_metadata:
    print(f'\n\nRunning analysis for {model_name}')
    metrics, LaBo_metrics = run_analysis_lf(lf_models_metadata[model_name])
    output_images(metrics, overwrite=True)
    output_images(LaBo_metrics, overwrite=True)
    latex_table = print_results_latex(metrics,latex_table)
    print(LaBo_metrics)
    latex_table = print_labo_results_latex(LaBo_metrics,latex_table)
    all_metrics_LF.append(metrics)
    
for model_name in models_metadata:
    print(f'\n\nRunning analysis for {model_name}')
    metrics = load_metrics(models_metadata[model_name], exclude={})
    
    '''
                           exclude={'seed': [88000,
                                                                          88001,
                                                                          88002,
                                                                          88003,
                                                                          88004,
                                                                          88010,
                                                                          88011,
                                                                          88012,
                                                                          88013,
                                                                          88014,
                                                                          88020,
                                                                          88021,
                                                                          88022,
                                                                          88023,
                                                                          88024,
                                                                          88030,
                                                                          88031,
                                                                          88032,
                                                                          88033,
                                                                          88034]})
    '''
                                              
    if metrics is None:
        continue
    latex_table = print_results_latex(metrics,latex_table)
    output_images(metrics, overwrite=True)
    all_metrics.append(metrics)
    



print(latex_table)