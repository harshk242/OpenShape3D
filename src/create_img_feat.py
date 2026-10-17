import torch
import sys
import os 

from param import parse_args
from utils.misc import load_config, dump_config
from models.LogitScaleNetwork import LogitScaleNetwork
import models
from modelnet import make_modelnet40test, test_modelnet40
from objaverse_lvis import make_objaverse_lvis
import data
from tqdm import tqdm
import numpy as np

from utils_func import load_model, load_model_from_path 

torch.manual_seed(2024)

cli_args, extras = parse_args(sys.argv[1:])
config = load_config(cli_args.config, cli_args = vars(cli_args), extra_args = extras)
# device = torch.device("cuda:0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

objaverse_lvis_loader = make_objaverse_lvis(config)
all_image_feat = []
idx = 0

print("objaverse_lvis_loader size: ", len(objaverse_lvis_loader))
print("objaverse_lvis_loader dataset size: ", len(objaverse_lvis_loader.dataset))

with torch.no_grad():
    for data in tqdm(objaverse_lvis_loader):
        if idx > -1:
            image_feat = data["image_feat"]
            all_image_feat.append(image_feat)
            idx += 1
        else:
            break
all_image_feat = np.concatenate(all_image_feat, axis=0)
print("all_image_feat: ", all_image_feat.shape)
np.save("meta_data/all_image_feat.npy", all_image_feat)
