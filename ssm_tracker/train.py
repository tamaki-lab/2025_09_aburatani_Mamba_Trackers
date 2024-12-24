import os 
import numpy as np 
import torch 
import yaml 
import argparse

from dataset.dataset import TrajDataset, TrajDatasetv2
from torch.utils.data import DataLoader
from train_utils.envs import select_device
from train_utils.lr_scheduler import TransformerLRScheduler, NoneLRScheduler

from models.MambaTrack import MambaTrack
from models.TrackSSM import TrackSSM

from loguru import logger
from tqdm import tqdm


MODEL_GALLERY = {
    'MambaTrack': MambaTrack, 
    'TrackSSM': TrackSSM
}

def train(args):

    config_file = args.config_file
    with open(config_file, 'r') as f:
        cfgs = yaml.safe_load(f)
    dataset_cfgs = cfgs['dataset']
    train_cfgs = cfgs['train']

    # copy the experiment config file to the save path 
    save_dir = os.path.join(args.save_path, args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    os.system(f'cp {config_file} {save_dir}')

    # dataset 
    logger.info(f'Loading data')
    dataset = TrajDatasetv2(dataset_cfgs)

    dataloader = DataLoader(dataset, 
                            batch_size=train_cfgs['batch_size'], 
                            shuffle=True, 
                            )

    # model
    logger.info('Loading model')
    model = MODEL_GALLERY[train_cfgs['model']](train_cfgs)
    device = select_device(args.device)
    model.to(device)
    model.train()

    # optimizer
    if train_cfgs['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params=filter(lambda x : x.requires_grad, model.parameters()), 
                                    lr=train_cfgs['lr0'], 
                                    )
    elif train_cfgs['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params=filter(lambda x : x.requires_grad, model.parameters()), 
                                     lr=train_cfgs['lr0'], 
                                     betas=[0.9, 0.98], eps=1e-8)
    else:
        optimizer = None
        
    # lr scheduler
    if train_cfgs['lr_scheduler'] == 'transformer':
        lr_scheduler = TransformerLRScheduler(optimizer=optimizer, d_model=train_cfgs['d_m'], warmup_steps=4000)
    elif train_cfgs['lr_scheduler'] == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.5, total_iters=10)
    else:
        lr_scheduler = NoneLRScheduler(lr=train_cfgs['lr0'])

    # if resume training, load the ckpt 
    if args.resume:
        load_ckpt = torch.load(args.resume_ckpt)
        model.load_state_dict(load_ckpt['model'])
        optimizer.load_state_dict(load_ckpt['optimizer'])
        start_epoch = load_ckpt['epoch'] + 1
        logger.info(f'Resume training from epoch {start_epoch}')
    else:
        start_epoch = 1

    # train
    logger.info('Start training!')
    
    for epoch in range(start_epoch, train_cfgs['epochs'] + 1):
        logger.info(f'Start epoch {epoch}')

        max_iter = len(dataloader)
        process_bar = tqdm(enumerate(dataloader), total=max_iter, ncols=150)

        model.train()
        optimizer.zero_grad()

        mean_loss = 0.0

        if not train_cfgs['lr_scheduler'] == 'transformer':
            lr_scheduler.step()  # most lr scheduler is updated per epoch

        for step, data in process_bar:
            if type(data) == dict:
                x = data['condition'].to(device)
                label = data['label'].to(device)
            else:
                x = data[0].to(device)
                label = data[1].to(device)

            loss = model.forward(x, label)

            loss_value = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if train_cfgs['lr_scheduler'] == 'transformer':
                lr_scheduler.step()  # transformer scheduler is updated per step

            cur_lr = optimizer.param_groups[0]['lr']


            mean_loss = (mean_loss * step + loss_value) / (step + 1)

            if step % 50 == 0:
                process_bar.set_description(f'epoch: {epoch}, iter: {step}, lr: {cur_lr:.8f}, current loss: {loss_value:.10f}, mean loss: {mean_loss:.10f}')

        if not epoch % train_cfgs['save_period']:
            # save model
            save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'epoch': epoch,
            }
                
            save_file = os.path.join(save_dir, f'epoch{epoch}.pth')
            torch.save(save_dict, save_file)
            logger.info(f'Saved ckpt file at {save_file}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mamba Tracker Training')

    parser.add_argument('--exp_name', type=str, default='mamba_track', help='experiment name for ckpt saving')
    parser.add_argument('--config_file', type=str, default='cfgs/MambaTrack.yaml')
    parser.add_argument('--device', type=str, default='4')
    parser.add_argument('--save_path', type=str, default='./saved_ckpts')

    # resume training
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_ckpt', type=str, default=None)

    args = parser.parse_args()
    train(args)