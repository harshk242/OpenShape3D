module load cuda/11.3
module load gcc/11.2
conda create -n openshape python=3.9
conda activate openshape
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# pip install -U git+https://github.com/NVIDIA/MinkowskiEngine
conda install -c dglteam/label/cu113 dgl
pip install huggingface_hub wandb omegaconf torch_redstone einops tqdm open3d
pip install open_clip_torch