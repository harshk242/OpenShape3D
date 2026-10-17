import torch
import sys
import os 

from param import parse_args
from utils.misc import load_config, dump_config
from models.LogitScaleNetwork import LogitScaleNetwork
import models
from modelnet import make_modelnet40test, test_modelnet40
from objaverse_lvis import make_objaverse_lvis, test_objaverse_lvis, xyz_objaverse_lvis, \
rgb_objaverse_lvis, features_objaverse_lvis
import data

from utils_func import load_model, load_model_from_path, load_model_no_classification

# def test_dataset(config, model, dataset, device, text_proj=None):

cli_args, extras = parse_args(sys.argv[1:])
# config = load_config("src/configs/test.yaml", cli_args = vars(cli_args), extra_args = extras)
config = load_config(cli_args.config, cli_args = vars(cli_args), extra_args = extras)
# device = torch.device("cuda:0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Using device: ", device)

torch.manual_seed(2024)

class EnsembleModel(torch.nn.Module):
    def __init__(self, base_model, base_proj_layer, finetune_proj_layer, w1, w2):
        super(EnsembleModel, self).__init__()
        self.base_model = base_model
        self.base_proj_layer = base_proj_layer
        self.finetune_proj_layer = finetune_proj_layer
        self.w1 = torch.nn.Parameter(torch.Tensor([w1]), requires_grad=False)
        self.w2 = torch.nn.Parameter(torch.Tensor([w2]), requires_grad=False)

        # Freeze the layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.base_proj_layer.parameters():
            param.requires_grad = False

        for param in self.finetune_proj_layer.parameters():
            param.requires_grad = False

    def forward(self, xyz, features):
        # xyz, features = input_tuple
        x = self.base_model(xyz, features)
        base = self.base_proj_layer(x)
        finetune = self.finetune_proj_layer(x)
        res = self.w1 * base + self.w2 * finetune
        return res

model_name = "OpenShape/openshape-pointbert-vitg14-rgb"
base_model = load_model(config, model_name=model_name)
base_model_no_classification = load_model_no_classification(config, model_name=model_name)
# base_model.to(device)

input_channels = None
output_channels = None
base_proj_layer = None

for name, module in base_model.named_modules():

    # Check if the module is an instance of torch.nn.Linear
    if name.find("proj") != -1:
        base_proj_layer = module

base_model = None
# base_model_layers = list(base_model.children())
# base_model_layers = base_model_layers[:-1]
# base_model = torch.nn.Sequential(*base_model_layers).to(device)

for name, param in base_proj_layer.named_parameters():
    param.requires_grad = False


if config.resume is not None:
    finetune_model = load_model_from_path(config, config.resume, device)

finetune_proj_layer = None
for name, module in finetune_model.named_modules():
    if name.find("proj") != -1:
        finetune_proj_layer = module

finetune_model = None

base_model_no_classification.eval()
base_proj_layer.eval()
finetune_proj_layer.eval()

w2 = 0.7 # weight to finetuned model
ensemble_model = EnsembleModel(base_model_no_classification, base_proj_layer, finetune_proj_layer, 1-w2, w2)
ensemble_model.to(device)
ensemble_model.eval()

logit_scale = LogitScaleNetwork(config.training.logit_scale_init).to(device)
image_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(device)
text_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(device)

# objaverse_lvis_loader = make_objaverse_lvis(config)

# phase = "all"
phase = "test"
cat_list = ['goldfish', 'fish', 'salmon_(fish)', 'earphone', 'headset', 'blazer', 'jacket', 'teakettle', 'teapot', 'kettle']
finetune_test_loader = data.make(config, phase, None, None, cat_list)

print(f"finetune_test_loader batches: {len(finetune_test_loader)}")
print(f"finetune_test_loader dataset size: {len(finetune_test_loader.dataset)}")


# test_modelnet40(model, config, modelnet40_loader, text_proj, device)
# objaverse_dict = test_objaverse_lvis(ensemble_model, config, objaverse_lvis_loader, text_proj, device)
objaverse_dict = test_objaverse_lvis(ensemble_model, config, finetune_test_loader, text_proj, device)
