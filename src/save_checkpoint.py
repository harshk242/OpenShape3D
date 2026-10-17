import torch
import sys
from collections import OrderedDict
import re

import models
from param import parse_args
from utils.misc import load_config, dump_config
from utils_func import load_model, load_model_from_path, load_model_no_classification
from utils_func import hf_hub_download

cli_args, extras = parse_args(sys.argv[1:])
# config = load_config("src/configs/test.yaml", cli_args = vars(cli_args), extra_args = extras)
config = load_config(cli_args.config, cli_args = vars(cli_args), extra_args = extras)

model_name = "OpenShape/openshape-pointbert-vitg14-rgb"
checkpoint = torch.load(hf_hub_download(repo_id=model_name, filename="model.pt"))
model_dict = OrderedDict()
pattern = re.compile('module.')
for k,v in checkpoint['state_dict'].items():
    if re.search("module", k):
        model_dict[re.sub(pattern, '', k)] = v

# checkpoint = torch.load(path)
device = torch.device("cuda")
model = models.make(config).to(device)
model.load_state_dict(model_dict)

checkpoint['state_dict'] = model_dict

torch.save(checkpoint, "pretrained/openshape-pointbert-vitg14-rgb/checkpoint.pt")


print("Model loaded from checkpoint")
# print("Model: ", model)