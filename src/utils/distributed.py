import os
import torch.distributed as dist

def setup(rank, world_size, port='12356'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
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
