import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import random
import json

from utils.data import normalize_pc
from utils_func import calc_accuracy

class ObjaverseLVIS(Dataset):
    def __init__(self, config):
        self.split = json.load(open(config.objaverse_lvis.split, "r"))
        self.y_up = config.objaverse_lvis.y_up
        self.num_points = config.objaverse_lvis.num_points
        self.use_color = config.objaverse_lvis.use_color
        self.normalize = config.objaverse_lvis.normalize
        self.categories = sorted(np.unique([data['category'] for data in self.split]))
        self.category2idx = {self.categories[i]: i for i in range(len(self.categories))}
        self.clip_cat_feat = np.load(config.objaverse_lvis.clip_feat_path, allow_pickle=True)
        # logging.info("ObjaverseLVIS: %d samples" % (len(self.split)))
        # logging.info("----clip feature shape: %s" % str(self.clip_cat_feat.shape))

    def __getitem__(self, index: int):
        path = self.split[index]['data_path'][1:]
        data = np.load(path, allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]
        # image_feat = torch.from_numpy(data['image_feat'])


        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.use_color:
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz
        
        assert not np.isnan(xyz).any()

        if np.random.rand() < 0.5:
            image_feat = data['thumbnail_feat']
        else:
            idx = np.random.randint(data['image_feat'].shape[0])
            image_feat = data["image_feat"][idx]

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "rgb": torch.from_numpy(rgb).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "group": self.split[index]['group'],
            "name":  self.split[index]['uid'],
            "category": self.category2idx[self.split[index]["category"]],
            "image_feat": torch.from_numpy(image_feat)
        }

    def __len__(self):
        return len(self.split)
  

def objaverse_lvis_collate_fn(list_data):
    return {
        # "xyz": ME.utils.batched_coordinates([data["xyz"] for data in list_data], dtype=torch.float32),
        "features": torch.cat([data["features"] for data in list_data], dim=0),
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "rgb_dense": torch.stack([data["rgb"] for data in list_data]).float(),
        "features_dense": torch.stack([data["features"] for data in list_data]),
        "group": [data["group"] for data in list_data],
        "name": [data["name"] for data in list_data],
        "category": torch.tensor([data["category"] for data in list_data], dtype = torch.int32),
        "image_feat": torch.stack([data["image_feat"] for data in list_data])
    }


def make_objaverse_lvis(config):
    print("Loading ObjaverseLVIS...")
    return DataLoader(
        ObjaverseLVIS(config), \
        num_workers=config.objaverse_lvis.num_workers, \
        collate_fn=objaverse_lvis_collate_fn, \
        batch_size=config.objaverse_lvis.batch_size, \
        pin_memory = True, \
        shuffle=False 
    )

def test_objaverse_lvis(model, config, objaverse_lvis_loader, text_proj, device):
    model.eval()
    if config.training.use_text_proj:
        text_proj.eval()
    clip_text_feat = torch.from_numpy(objaverse_lvis_loader.dataset.clip_cat_feat).to(device)
    if config.training.use_text_proj:
        clip_text_feat = text_proj(clip_text_feat)
    per_cat_correct = torch.zeros(1156).to(device)
    per_cat_count = torch.zeros(1156).to(device)
    category2idx = objaverse_lvis_loader.dataset.category2idx
    idx2category = {v: k for k, v in category2idx.items()}
    
    logits_all = []
    labels_all = []
    data_xyz = []
    unique_labels = []

    # pred_feat:  torch.Size([70, 1280])
    # clip_text_feat:  torch.Size([1156, 1280])
    # logits:  torch.Size([70, 1156])
    # lables:  torch.Size([70])
    runs = 0
    with torch.no_grad():
        for data in tqdm(objaverse_lvis_loader):
            data['xyz_dense'] = data['xyz_dense'].to(device)
            data['features_dense'] = data['features_dense'].to(device)
            
            pred_feat = model(data['xyz_dense'], data['features_dense'])
            logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
            labels = data['category'].to(device)
            data_xyz.append(data['xyz_dense'])
            logits_all.append(logits.detach())
            labels_all.append(labels)
            unique_labels.append(torch.unique(labels))
            # calculate per class accuracy

            for i in torch.unique(labels):
                idx = (labels == i)
                if idx.sum() > 0:
                    per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
                    per_cat_count[i] += idx.sum()
            runs += 1
    unique_labels = torch.unique(torch.cat(unique_labels)).long()
    print("unique_labels shape: ", unique_labels.shape)
    
    topk_acc, correct = calc_accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1,3,5,))

    per_cat_correct = per_cat_correct[unique_labels]
    per_cat_count = per_cat_count[unique_labels]

    overall_acc = per_cat_correct.sum() / per_cat_count.sum()
    per_cat_acc = per_cat_correct / per_cat_count


    objaverse_dict = {}
    objaverse_dict["per_cat_acc"] = per_cat_acc.clone().detach().cpu()
    objaverse_dict["xyz"] = torch.cat(data_xyz).clone().detach().cpu()
    objaverse_dict["labels_all"] = torch.cat(labels_all).clone().detach().cpu()
    objaverse_dict["logits_all"] = torch.cat(logits_all).clone().detach().cpu()
    objaverse_dict["category2idx"] = category2idx
    objaverse_dict["idx2category"] = idx2category
    if config.export_results:
        print("Exporting ObjaverseLVIS results...")
        torch.save(objaverse_dict, "src/eval_data/objaverse_dict.pt")


    print('Test ObjaverseLVIS: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
    print('Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()))

    return objaverse_dict

def xyz_objaverse_lvis(objaverse_lvis_loader, device):
    
    data_xyz = []
    with torch.no_grad():
        for data in objaverse_lvis_loader:
          
            data_xyz.append(data['xyz_dense'])

    torch.save(torch.cat(data_xyz).clone().detach().cpu(), "src/eval_data/ov_xyz.pt")

def rgb_objaverse_lvis(objaverse_lvis_loader, device):
    
    rgb = []
    with torch.no_grad():
        for data in objaverse_lvis_loader:
          
            rgb.append(data['rgb_dense'])

    torch.save(torch.cat(rgb).clone().detach().cpu(), "src/eval_data/ov_rgb.pt")

def features_objaverse_lvis(objaverse_lvis_loader, device):
    
    features = []
    with torch.no_grad():
        for data in objaverse_lvis_loader:
          
            features.append(data['features_dense'])

    torch.save(torch.cat(features).clone().detach().cpu(), "src/eval_data/ov_features.pt")