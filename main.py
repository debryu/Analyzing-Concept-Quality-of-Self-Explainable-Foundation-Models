import sys, os
import torch
import argparse
import importlib
import setproctitle, socket, uuid
import datetime

from utils.osr import get_osr_params
from datasets import get_dataset
from models import get_model
from utils.train_from_scratch import train
from utils.evaluate import eval
from utils.conf import *
from utils.args import *
from utils.checkpoint import create_load_ckpt
from utils.utils import get_dataset_C_Y

from utils.train_cbm import train_cbm_and_save, train_LF_CBM, get_args

conf_path = os.getcwd() + "."
sys.path.append(conf_path)

parser = argparse.ArgumentParser(description='Interpreter LCBM', allow_abbrev=False)
parser.add_argument('--modality', type=str, default='custom', help='Select if training the model from scratch, if preprocessing CLIP, if learning from CLIP',
                    choices=['custom', 'preCLIP', 'postCLIP'])

def preprocess_with_CLIP():

    # load args and run preprocessing
    args_cbm = get_args()
    train_cbm_and_save(args_cbm)

def custom_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,default='cext', help='Model for inference.', choices=get_all_models())
    torch.set_num_threads(4)

    

    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    # LOAD THE PARSER SPECIFIC OF THE MODEL, WITH ITS SPECIFICS
    get_parser = getattr(mod, 'get_parser') 
    parser = get_parser()
    parser.add_argument('--project', type=str, default="thesis", help='wandb project')

    print(parser)

    args = parser.parse_args() # this is the return

    # load args related to seed etc.
    set_random_seed(args.seed) if args.seed is not None else set_random_seed(42)
    
    return args

def default_training(args):
    
    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    

    # Load dataset, model, loss, and optimizer
    args.num_C, args.num_Y = get_dataset_C_Y(args)
    dataset = get_dataset(args)
    encoder, decoder  = dataset.get_backbone(args)
    model = get_model(args, encoder, decoder) 
    loss  = model.get_loss(args)
    model.start_optim(args)

    # SAVE A BASE MODEL OR LOAD IT, LOAD A CHECKPOINT IF PROVIDED
    model = create_load_ckpt(model, args)

    # set job name
    setproctitle.setproctitle('{}_{}_{}'.format( args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))

    # perform posthoc evaluation/ cl training/ joint training
    print('--->    Chosen device: ', model.device, '\n')
    
    if args.posthoc: 
        #Run evaluation
        th_r, th_y, proto = get_osr_params(model, dataset, args)
        print(eval(model, dataset, loss, args, th_r, th_y, proto, plot = True))
        
    else: 
        train(model, dataset, loss, args)

    

if __name__=='__main__':

    # TODO: create a switch from parser
    modality = 'custom'
    # Start a timer to measure the time
    start = datetime.datetime.now()

    if modality == 'custom':
        args = custom_parse_args()
        print(args)
        default_training(args)
    
    elif modality == 'preCLIP':
        args = get_args()
        train_cbm_and_save(args)

    elif modality == 'postCLIP':
        args = get_args()
        train_LF_CBM(args)

    else:
        NotImplementedError('Invalid choice.')
    end = datetime.datetime.now()
    print('\n ### Total time taken: ', end - start)
    print('\n ### Closing ###')