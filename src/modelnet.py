import numpy as np
import torch
import torch.nn.functional as F
import random
import copy
import json
from torch.utils.data import Dataset, DataLoader

from utils.data import random_rotate_z, normalize_pc, augment_pc
from utils_func import calc_accuracy

cpu = torch.device("cpu")

class ModelNet40Test(Dataset):
    def __init__(self, config):
        self.split = json.load(open(config.modelnet40.test_split, "r"))
        self.pcs = np.load(config.modelnet40.test_pc, allow_pickle=True)
        self.num_points = config.modelnet40.num_points
        self.use_color = config.dataset.use_color
        self.y_up = config.modelnet40.y_up
        clip_feat = np.load(config.modelnet40.clip_feat_path, allow_pickle=True).item()
        self.categories = list(clip_feat.keys())
        self.clip_cat_feat = []
        self.category2idx = {}
        for i, category in enumerate(self.categories):
            self.category2idx[category] = i
            self.clip_cat_feat.append(clip_feat[category]["open_clip_text_feat"])
        self.clip_cat_feat = np.concatenate(self.clip_cat_feat, axis=0)
        # logging.info("ModelNet40Test: %d samples" % len(self.split))
        # logging.info("----clip feature shape: %s" % str(self.clip_cat_feat.shape))

    def __getitem__(self, index: int):
        pc = copy.deepcopy(self.pcs[index])
        n = pc['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = pc['xyz'][idx]
        rgb = pc['rgb'][idx]
        rgb = rgb / 255.0 # 100, scale to 0.4 to make it consistent with the training data
        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        
        xyz = normalize_pc(xyz)

        if self.use_color:
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz
        
        assert not np.isnan(xyz).any()
        
        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "name": self.split[index]["name"],
            "category": self.category2idx[self.split[index]["category"]],
            # "description": self.split[index]["category"]
        }

    def __len__(self):
        return len(self.split)

def modelnet40_collate_fn(list_data):
    return {
        # "xyz": ME.utils.batched_coordinates([data["xyz"] for data in list_data], dtype=torch.float32),
        # "xyz": torch.Tensor([data["xyz"] for data in list_data], dtype=torch.float32),
        "xyz": torch.stack([data["xyz"] for data in list_data]).float(),
        "features": torch.cat([data["features"] for data in list_data], dim=0),
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "features_dense": torch.stack([data["features"] for data in list_data]),
        "name": [data["name"] for data in list_data],
        "category": torch.tensor([data["category"] for data in list_data], dtype = torch.int32),
    }

def make_modelnet40test(config):
    print("Loading ModelNet40...")
    dataset = ModelNet40Test(config)
    data_loader = DataLoader(
        dataset, \
        num_workers=config.modelnet40.num_workers, \
        collate_fn=modelnet40_collate_fn, \
        batch_size=config.modelnet40.test_batch_size, \
        pin_memory = True, \
        shuffle=False
    )
    return data_loader

def test_modelnet40(model, config, modelnet40_loader, text_proj, device):
    model.eval()
    if config.training.use_text_proj:
        text_proj.eval()
    clip_text_feat = torch.from_numpy(modelnet40_loader.dataset.clip_cat_feat).to(device)
    if config.training.use_text_proj:
        clip_text_feat = text_proj(clip_text_feat)
    per_cat_correct = torch.zeros(40).to(device)
    per_cat_count = torch.zeros(40).to(device)
    category2idx = modelnet40_loader.dataset.category2idx
    idx2category = {v: k for k, v in category2idx.items()}
    
    logits_all = []
    labels_all = []
    data_xyz = []
    with torch.no_grad():
        for data in modelnet40_loader:
            data['xyz_dense'] = data['xyz_dense'].to(device)
            data['features_dense'] = data['features_dense'].to(device)
            
            pred_feat = model(data['xyz_dense'], data['features_dense'], device = device)
            logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
            labels = data['category'].to(device)
            data_xyz.append(data['xyz_dense'])
            logits_all.append(logits.detach())
            labels_all.append(labels)

            for i in range(40):
                idx = (labels == i)
                if idx.sum() > 0:
                    per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
                    per_cat_count[i] += idx.sum()
    topk_acc, correct = calc_accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1,3,5,))

    overall_acc = per_cat_correct.sum() / per_cat_count.sum()
    per_cat_acc = per_cat_correct / per_cat_count
    modelnet_dict = {}

    modelnet_dict["per_cat_acc"] = per_cat_acc.clone().detach().cpu()
    modelnet_dict["data_xyz"] = torch.cat(data_xyz).clone().detach().cpu()
    modelnet_dict["labels_all"] = torch.cat(labels_all).clone().detach().cpu()
    modelnet_dict["logits_all"] = torch.cat(logits_all).clone().detach().cpu()
    modelnet_dict["category2idx"] = category2idx
    modelnet_dict["idx2category"] = idx2category

    torch.save(modelnet_dict, "src/eval_data/modelnet_dict.pt")


    eval_dict = {}
    for i in torch.argsort(per_cat_acc):
        eval_dict[idx2category[i.item()]] = {"acc": per_cat_acc[i].item(), "count": per_cat_count[i].item(), 
                                             "correct": per_cat_correct[i].item()}
        
    
    #for i in range(40):


    print('Test ModelNet40: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
    print('Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()))
