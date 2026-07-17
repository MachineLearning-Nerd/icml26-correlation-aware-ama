import logging
import torch
import os
import json
from argparse import Namespace

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def save_model(model, file_path, epoch, args):
    # os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'args': vars(args) if hasattr(args, '__dict__') else args
    }
    
    torch.save(state, f"{file_path}_{epoch}.pt")
    
    with open(f"{file_path}_args.json", 'w') as f:
        json.dump(state['args'], f, indent=2)
        
def load_model(model_class, model_path, device='auto'):
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
        
    checkpoint = torch.load(model_path, map_location=device)
    
    saved_args_dict = checkpoint['args']
    saved_args = Namespace(**saved_args_dict)
    saved_args.device = device

    model = model_class(saved_args)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    # model.eval()
    
    return model, checkpoint