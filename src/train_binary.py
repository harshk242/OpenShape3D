import torch
import time
import numpy as np
import wandb
import logging
import os
import re
import torch.distributed.nn
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from collections import OrderedDict

import sklearn.metrics as metrics
from utils_func import hf_hub_download

def get_new_labels(labels, cat_list):
    label_new = torch.zeros_like(labels)
    for idx, cat in enumerate(cat_list):
        label_new[labels == cat] = idx
    label_new = label_new.float()
    
    return label_new

class Trainer(object):
    def __init__(self, rank, config, model, optimizer, scheduler, criterion, train_loader, test_loader, cat_list):
        
        self.rank = rank
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        category2idx = train_loader.dataset.category2idx
        self.cat_list = [category2idx[cat] for cat in cat_list]
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epoch = 0
        self.best_epoch = 0
        self.step = 0
        self.best_acc = 0

        
        if self.config.text_model == "vico":
            self.vico_feat_dict = np.load(config.vico.dict_path, allow_pickle=True).item()

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_img_contras_acc = checkpoint['best_img_contras_acc']
        self.best_text_contras_acc = checkpoint['best_text_contras_acc']
        self.best_modelnet40_overall_acc = checkpoint['best_modelnet40_overall_acc']
        self.best_modelnet40_class_acc = checkpoint['best_modelnet40_class_acc']
        self.best_lvis_acc = checkpoint['best_lvis_acc']

        logging.info("Loaded checkpoint from {}".format(path))
        logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))
        logging.info("----Best img contras acc: {}".format(self.best_img_contras_acc))
        logging.info("----Best text contras acc: {}".format(self.best_text_contras_acc))
        logging.info("----Best modelnet40 overall acc: {}".format(self.best_modelnet40_overall_acc))
        logging.info("----Best modelnet40 class acc: {}".format(self.best_modelnet40_class_acc))
        logging.info("----Best lvis acc: {}".format(self.best_lvis_acc))

    def train_one_epoch(self):
        self.model.train()

        for data in self.train_loader:
            self.step += 1
            self.optimizer.zero_grad()
            loss = 0
            data['xyz'] = data['xyz'].to(self.config.device)
            data['features'] = data['features'].to(self.config.device)

            logits = self.model(data['xyz'], data['features'])
            logits = logits.squeeze()
            labels = data['category'].to(self.config.device)

            label_new = get_new_labels(labels, self.cat_list)

            loss = self.criterion(logits, label_new)
            loss.backward()

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            if self.rank == 0 and self.step % self.config.training.log_freq == 0:
                logging.info("Epoch: {} Step: {} Loss: {}".format(self.epoch, self.step, loss.item()))

                if self.config.wandb_key is not None:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": self.optimizer.param_groups[0]['lr'],
                        "train/epoch": self.epoch,
                        "train/step": self.step
                    })

    def save_model(self, name):
        torch_dict = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.epoch,
            "step": self.step,
        }

        torch.save(torch_dict, os.path.join(self.config.ckpt_dir, '{}.pt'.format(name)))

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res, correct

    def train(self):
        for epoch in range(self.epoch, self.config.training.max_epoch):
            self.epoch = epoch
            if self.rank == 0 and self.step % self.config.training.log_freq == 0:
                logging.info("Epoch: {}".format(self.epoch))

            start_time = time.time()
            self.train_one_epoch()
            time_per_epoch = int(time.time() - start_time)

            formatted_time = "{:02}:{:02}:{:02}".format(
                divmod(time_per_epoch, 3600)[0],
                divmod(time_per_epoch, 60)[0] % 60,
                time_per_epoch % 60,
            )
            if self.rank == 0:
                logging.info(f"Time taken for epoch {epoch}: {formatted_time}")
            
            overall_acc = self.test()
            if overall_acc > self.best_acc:
                self.best_acc = overall_acc
                self.best_epoch = self.epoch
                self.save_model('best')
                logging.info("Best acc: {}".format(self.best_acc))

            if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
                self.save_model('epoch_{}'.format(self.epoch))

        return self.best_acc, self.best_epoch

    def test(self):
        self.model.eval()
        
        logits_all = []
        labels_all = []
        preds_all = []
        with torch.no_grad():
            for data in self.test_loader:
                
                # logits = self.model(data['xyz'], data['features']) data['xyz_dense'], data['features_dense']
                logits = self.model(data['xyz_dense'], data['features_dense'])
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                # labels_all.append(labels)

                labels_new = get_new_labels(labels, self.cat_list)
                
                sigmoid = nn.Sigmoid()
                logits = sigmoid(logits)
                preds = logits > 0.5
                
                labels_all.append(labels_new.cpu().numpy())
                preds_all.append(preds.detach().cpu().numpy())
        
        labels_all = np.concatenate(labels_all)
        preds_all = np.concatenate(preds_all).reshape(-1)

        overall_acc = metrics.accuracy_score(labels_all, preds_all)

        if overall_acc > self.best_acc:
            self.best_acc = overall_acc
            self.save_model('best')
        
        if self.rank == 0 and self.step % self.config.training.log_freq == 0:
            logging.info(f"Test ObjaverseLVIS: overall acc: {overall_acc}")
        
        if self.config.wandb_key is not None:
            wandb.log({"test/epoch": self.epoch,
                    "test/step": self.step,
                    "test/overall_acc": overall_acc,
                    })
        
        return overall_acc
