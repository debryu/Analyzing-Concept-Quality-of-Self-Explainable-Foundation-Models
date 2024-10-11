
import os
import importlib

def get_all_models():
    return [model.split('.')[0] for model in os.listdir('models')
            if not model.find('__') > -1 and 'py' in model]
names = {}
for model in get_all_models():
    if model == 'cbm': continue
    mod = importlib.import_module('models.' + model)
    class_name = {x.lower():x for x in mod.__dir__()}[model.replace('_', '')]
    names[model] = getattr(mod, class_name)

def get_model(args, encoder, decoder):
    if args.model in ['cext']:
        return names[args.model](encoder, args=args) # only discriminative
    if args.model in ['cvae', 'conceptvae']:
        if args.dataset in ['mnist']:
            return names[args.model](encoder, decoder, c_split = (args.num_C,), args=args) # add concept splits [2 for in-distribution, 2 for out-of-distribution]
        if args.dataset in ['shapes3d']:
            return names[args.model](encoder, decoder, c_split = (10,10,10,8,4), args=args)
    if args.model in ['betaglancenet']:
        return names[args.model](encoder, decoder, args=args)
    if args.model in ['blackbox']:
        return names[args.model](encoder, decoder, args.num_C, args=args)
    else:
        return names[args.model](encoder, decoder, args=args)