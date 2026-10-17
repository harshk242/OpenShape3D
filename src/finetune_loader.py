import os
import json
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
import logging

from utils.data import normalize_pc

rando_seed = 2020

class FinetuneLoader(Dataset):
    def __init__(self, config, cat_list, split="train", train_split=0.8):
        random.seed(rando_seed)

        objaverse_dict = torch.load("./src/eval_data/objaverse_dict.pt")
        ov_category2idx = objaverse_dict["category2idx"]
        ov_idx2category = objaverse_dict["idx2category"]
        objaverse_dict = None

        lvis_test_file = config.objaverse_lvis.split
        lvis_test = json.load(open(lvis_test_file, "r"))

        cat_idx_list = [ov_category2idx[cat] for cat in cat_list]
        logging.info("cat_idx_list: {}".format(cat_idx_list))

        filter_data = {}
        for i in cat_idx_list:
            filter_data[i] = []

        for obj in lvis_test:
            cat_id = ov_category2idx[obj["category"]]
            if cat_id in cat_idx_list:
                filter_data[cat_id].append(obj)

        train_data = []
        test_data = []

        for key, val in filter_data.items():
            # logging.info(f"Category: {ov_idx2category[key]}, Size: {len(val)}")
            shuffle_val = val.copy()
            
            data_size = len(val)
            random.shuffle(shuffle_val)
            train_size = int(train_split * data_size)
            train_data.extend(shuffle_val[:train_size])
            test_data.extend(shuffle_val[train_size:])

        self.y_up = config.dataset.y_up
        self.normalize = config.dataset.normalize
        self.use_color = config.dataset.use_color
        self.num_points = config.objaverse_lvis.num_points
        self.text_source = config.dataset.text_source
        self.use_text_filtering = config.dataset.use_text_filtering
        self.use_prompt_engineering = config.dataset.use_prompt_engineering

        self.objaverse_lvis_split = json.load(open(config.objaverse_lvis.split, "r"))
        self.categories = sorted(np.unique([data['category'] for data in self.objaverse_lvis_split]))
        self.category2idx = {self.categories[i]: i for i in range(len(self.categories))}

        self.clip_cat_feat = np.load(config.objaverse_lvis.clip_feat_path, allow_pickle=True)
        self.text_embed_version = "prompt_avg" if self.use_prompt_engineering else "original"
        if self.use_text_filtering:
            self.gpt4_filtering = json.load(open(config.dataset.gpt4_filtering_path, "r"))

        if split == "train":
            self.data = np.array(train_data)

        elif split == "test":
            self.data = np.array(test_data)

        elif split == "all":
            self.data = np.array(train_data + test_data)
            assert len(self.data) == len(train_data) + len(test_data)
        else:
            raise NotImplementedError

    def __getitem__(self, index: int):
        meta = self.data[index]
        data_path = meta["data_path"][1:]
        uid = meta["uid"]

        raw_data = np.load(data_path, allow_pickle=True).item()
        
        n = raw_data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = raw_data['xyz'][idx]
        rgb = raw_data['rgb'][idx]

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

        text_feat = []
        texts = []
        if 'text' in self.text_source:
            if not (self.use_text_filtering and self.gpt4_filtering[uid]["flag"] == "N"):
                texts.append(raw_data["text"][0])
                text_feat.append(raw_data["text_feat"][0][self.text_embed_version])
        
        if 'caption' in self.text_source:
            if np.random.rand() < 0.5:
                if len(raw_data["blip_caption"]) > 0:
                    texts.append(raw_data["blip_caption"])
                    text_feat.append(raw_data["blip_caption_feat"][self.text_embed_version])
            else:
                if len(raw_data["msft_caption"]) > 0:
                    texts.append(raw_data["msft_caption"])
                    text_feat.append(raw_data["msft_caption_feat"][self.text_embed_version])
        
        if 'retrieval_text' in self.text_source:
            if len(raw_data["retrieval_text"]) > 0:
                idx = np.random.randint(len(raw_data["retrieval_text"]))
                texts.append(raw_data["retrieval_text"][idx])
                text_feat.append(raw_data["retrieval_text_feat"][idx]["original"]) # no prompt engineering for retrieval text

        if len(text_feat) > 0:
            assert len(text_feat) == len(texts)
            text_idx = np.random.randint(len(texts))
            text_feat = text_feat[text_idx]
            texts = texts[text_idx]
            text_feat = torch.from_numpy(text_feat).type(torch.float32).reshape(-1)
        else:
            text_feat = None
            texts = None

        if np.random.rand() < 0.5:
            img_feat = raw_data['thumbnail_feat']
        else:
            idx = np.random.randint(raw_data['image_feat'].shape[0])
            img_feat = raw_data["image_feat"][idx]

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "img_feat": torch.from_numpy(img_feat).type(torch.float32).reshape(-1),
            "dataset": "Objaverse",
            "data_path": meta["data_path"],
            "group": meta["group"],
            "name":  uid,
            "category": meta["category"],
            "category2idx": self.category2idx[meta["category"]],
            "texts": texts,
            "text_feat": text_feat,
            "has_text": text_feat is not None,
        }

    def __len__(self):
        return len(self.data)
