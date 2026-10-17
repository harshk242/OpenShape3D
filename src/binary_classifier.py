import sys
import os
import logging
import shutil
import data
import models
import torch
import wandb
from torch import nn
from omegaconf import OmegaConf
from datetime import datetime
from param import parse_args
from utils.misc import load_config, dump_config    
from utils.logger import setup_logging
from train_binary import Trainer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from utils_func import load_model


cat_list = ["armor", "helmet"]

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def setup_cache(config):
    if config.cache_dir is not None:
        print("Setting cache: ", config.cache_dir)
        os.environ['TORCH_HOME'] = os.path.join(config.cache_dir, 'torch')
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(config.cache_dir, 'huggingface', 'transformers')
        os.environ['HF_DATASETS_CACHE'] = os.path.join(config.cache_dir, 'huggingface', 'datasets')
        os.environ['HF_HOME'] = os.path.join(config.cache_dir, 'huggingface', 'home')

def cleanup():
    dist.destroy_process_group()

class BinaryClassifier(torch.nn.Module):
    def __init__(self, base_model, clip_cat_feat, device):
        super(BinaryClassifier, self).__init__()
        self.base_model = base_model
        self.fc1 = torch.nn.Linear(1280, 1)

        self.clip_text_feat = torch.from_numpy(clip_cat_feat).to(device) # [1156, 1280]

        # Freeze the layers
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, xyz, features):
        x = self.base_model(xyz, features)

        x = self.fc1(x)

        return x

def binary_main(rank, world_size, cli_args, extras, *args):
    cat_list = args[0]
    setup(rank, world_size)
    config = load_config(cli_args.config, cli_args = vars(cli_args), extra_args = extras)

    setup_cache(config)

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

    train_loader = data.make(config, 'train', rank, world_size, cat_list)

    if rank == 0:
        test_loader = data.make(config, 'test', rank, world_size, cat_list)
    else:
        test_loader = None
    

    model_name = "OpenShape/openshape-pointbert-vitg14-rgb"
    base_model = load_model(config, model_name)
    model = BinaryClassifier(base_model, train_loader.dataset.clip_cat_feat, config.device).to(config.device)

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    logging.info("Using SyncBatchNorm")

    logging.info("Train loader size: {}".format(len(train_loader)))
    logging.info("Train loader dataset size: {}".format(len(train_loader.dataset)))

    logging.info("Test loader size: {}".format(len(test_loader)))
    logging.info("Test loader dataset size: {}".format(len(test_loader.dataset)))

    params = list(model.module.fc1.parameters())

    if config.training.use_openclip_optimizer_scheduler:
        optimizer = torch.optim.AdamW(
            params,
            lr=config.training.lr,
            betas=(config.training.beta1, config.training.beta2),
            eps=config.training.eps,
        )
    else:
        optimizer = torch.optim.AdamW(params, lr=config.training.lr)
    
    criterion = nn.BCEWithLogitsLoss()

    trainer = Trainer(rank, config, model, optimizer, None, criterion, train_loader, test_loader, cat_list)
    trainer.train()

    if rank == 0 and config.wandb_key is not None:
        wandb.finish()
    cleanup()


if __name__ == '__main__':
    cli_args, extras = parse_args(sys.argv[1:])
    
    world_size = cli_args.ngpu
    mp.spawn(
        binary_main,
        args=(world_size, cli_args, extras, cat_list),
        nprocs=world_size
    )
