import sys
import os
import logging
import shutil
import data
import models
import torch
import wandb
from omegaconf import OmegaConf
from datetime import datetime
from param import parse_args
from utils.misc import load_config, dump_config    
from utils.logger import setup_logging
from utils.distributed import setup, setup_cache, cleanup
from utils.scheduler import cosine_lr
from utils.load_model import load_from_local_checkpoint
from train import Trainer
from models.LogitScaleNetwork import LogitScaleNetwork
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from minlora import add_lora, get_lora_params
from utils_func import load_model


def main(rank, world_size, cli_args, extras):
    setup(rank, world_size, '12355')

    config = load_config(cli_args.config, cli_args = vars(cli_args), extra_args = extras)
    setup_cache(config)

    if config.autoresume:
        config.trial_name = config.get('trial_name') + "@autoresume"
    else:
        config.trial_name = config.get('trial_name') + datetime.now().strftime('@%Y%m%d-%H%M%S')
    config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
    config.code_dir = config.get('code_dir') or os.path.join(config.exp_dir, config.trial_name, 'code')

    if rank == 0:
        os.makedirs(os.path.join(config.exp_dir, config.trial_name), exist_ok=config.autoresume)
        os.makedirs(config.ckpt_dir, exist_ok=True)
        if os.path.exists(config.code_dir):
            shutil.rmtree(config.code_dir)
        shutil.copytree("./src", config.code_dir)
    
    config.device = 'cuda:{0}'.format(rank) 
    
    if rank == 0:
        config.log_path = config.get('log_path') or os.path.join(config.exp_dir, config.trial_name, 'log.txt')
        config.log_level = logging.DEBUG if config.debug else logging.INFO
        setup_logging(config.log_path, config.log_level)
        dump_config(os.path.join(config.exp_dir, config.trial_name, 'config.yaml'), config)
        logging.info("Using {} GPU(s).".format(config.ngpu))
        if config.wandb_key is not None:
            wandb.login(key=config.wandb_key)
            wandb.init(project=config.project_name, name=config.trial_name, config=OmegaConf.to_object(config))

    if config.train:
        model = models.make(config).to(config.device)
        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            logging.info(model)
            logging.info("Network:{}, Number of parameters: {}".format(config.model.name, total_params))
        
        model = load_from_local_checkpoint(model)
        add_lora(model)

        torch.cuda.set_device(rank)
        model.cuda(rank)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info("Using SyncBatchNorm")

        logit_scale = LogitScaleNetwork(config.training.logit_scale_init).to(config.device)
        image_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(config.device)
        text_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(config.device)

        logit_scale = DDP(logit_scale, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        image_proj = DDP(image_proj, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        text_proj = DDP(text_proj, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        train_loader = data.make(config, 'train', rank, world_size)
        logging.info("train_loader size: {}".format(len(train_loader)))
        logging.info("train_loader dataset size: {}".format(len(train_loader.dataset)))

        if rank == 0:
            modelnet40_loader = data.make_modelnet40test(config)
            objaverse_lvis_loader = data.make_objaverse_lvis(config)
            scanobjectnn_loader = data.make_scanobjectnntest(config)
            logging.info("objaverse_lvis_loader size: {}".format(len(objaverse_lvis_loader)))
            logging.info("objaverse_lvis_loader dataset size: {}".format(len(objaverse_lvis_loader.dataset)))
        else:
            modelnet40_loader = None
            objaverse_lvis_loader = None
            scanobjectnn_loader = None

        if rank == 0:
            if train_loader is not None:
                logging.info("Train iterations: {}".format(len(train_loader)))

        logging.info("lora params: {}".format(len(list(get_lora_params(model)))))
        params = list(get_lora_params(model)) + list(image_proj.parameters()) + list(text_proj.parameters()) + list(logit_scale.parameters()) 
        if config.training.use_openclip_optimizer_scheduler:
            optimizer = torch.optim.AdamW(
                params,
                lr=config.training.lr,
                betas=(config.training.beta1, config.training.beta2),
                eps=config.training.eps,
            )
            scheduler = cosine_lr(optimizer, config.training.lr, config.training.warmup, len(train_loader) * 16)
        else:
            optimizer = torch.optim.AdamW(params, lr=config.training.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.training.lr_decay_step, gamma=config.training.lr_decay_rate)


        trainer = Trainer(rank, config, model, logit_scale, image_proj, text_proj, optimizer, scheduler, \
                        train_loader, modelnet40_loader, objaverse_lvis_loader, scanobjectnn_loader, True)


        if config.resume is not None:
            trainer.load_from_checkpoint_lora(config.resume)

        trainer.train()

    if rank == 0 and config.wandb_key is not None:
        wandb.finish()
    cleanup()


if __name__ == '__main__':
    cli_args, extras = parse_args(sys.argv[1:])
    world_size = cli_args.ngpu
    mp.spawn(
        main,
        args=(world_size, cli_args, extras),
        nprocs=world_size
    )