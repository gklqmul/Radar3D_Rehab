import os
import torch
import pytorch_lightning as pl
import numpy as np
import logging

from sklearn.metrics import confusion_matrix
import seaborn as sns


from .wrapper import ModelWrapper
from .visualize import make_video


def get_checkpoint_file(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".ckpt"):
            return file


def plt_model_load(model, checkpoint):
    state_dict = torch.load(checkpoint)['state_dict']
    model.load_state_dict(state_dict)
    return model

def compute_confusion_matrix(model, dataloader, device='cuda', save_path="confusion_matrix"):
    model.eval() 
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch 
            inputs = inputs.to(device)
            labels = labels.to(device)
            model = model.to(device)
            outputs = model(inputs)  
            preds = torch.argmax(outputs, dim=1) 
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

   
    cm = confusion_matrix(all_labels, all_preds)
    
    np.savetxt(f"{save_path}_confusion.txt", cm, fmt="%d")
    
    return cm


def test(model, test_loader, plt_trainer_args, load_path, visualize):
    plt_model = ModelWrapper(model)
    if load_path is not None:
        if load_path.endswith(".ckpt"):
            checkpoint = load_path
        else:
            if load_path.endswith("/"):
                checkpoint = load_path + "best.ckpt"
            else:
                raise ValueError(
                    "if it is a directory, if must end with /; if it is a file, it must end with .ckpt")
        plt_model = plt_model_load(plt_model, checkpoint)
        plt_model.eval()
        print(f"Loaded model from {checkpoint}")

    trainer = pl.Trainer(**plt_trainer_args)
    trainer.test(plt_model, test_loader)

    compute_confusion_matrix(plt_model, test_loader, save_path=load_path)

    if visualize:
        filename = checkpoint[:-5]+'.avi'
        logging.info(f'Saving test result in {filename}')
        res = trainer.predict(plt_model, test_loader)
        Y_pred = np.concatenate([r[0] for r in res])
        Y = np.concatenate([r[1] for r in res])
        make_video(Y_pred, Y, filename)
        logging.info(f'Saved {filename}')