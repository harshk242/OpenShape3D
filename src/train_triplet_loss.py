import torch
import time
import numpy as np
import wandb
import logging
import os
import re
import torch.distributed.nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict
from sklearn.neighbors import NearestNeighbors

from utils_func import hf_hub_download


class TripletTrainer(object):
    def __init__(self, rank, config, model, logit_scale, image_proj, text_proj, optimizer, scheduler, train_loader, \
                  modelnet40_loader, objaverse_lvis_loader=None, scanobjectnn_loader=None, lora_used=False):
        self.rank = rank
        self.config = config
        self.model = model
        self.logit_scale = logit_scale
        self.image_proj = image_proj
        self.text_proj = text_proj
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.modelnet40_loader = modelnet40_loader
        self.objaverse_lvis_loader = objaverse_lvis_loader
        self.scanobjectnn_loader = scanobjectnn_loader
        self.epoch = 0
        self.step = 0
        self.best_img_contras_acc = 0
        self.best_text_contras_acc = 0
        self.best_modelnet40_overall_acc = 0
        self.best_modelnet40_class_acc = 0
        self.best_lvis_acc = 0
        self.lora_used = lora_used
        if self.config.text_model == "vico":
            self.vico_feat_dict = np.load(config.vico.dict_path, allow_pickle=True).item()
        
        self.lvis_eval_text_feat = np.load(config.objaverse_lvis.clip_feat_path, allow_pickle=True)
        self.lvis_eval_img_feat = np.load(config.objaverse_lvis.img_feat_path, allow_pickle=True)
        self.text_feat_nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(self.lvis_eval_text_feat)
        self.img_feat_nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(self.lvis_eval_img_feat)

        logging.info(f"Contras loss weight: {self.config.training.contras_loss_weight}")
        logging.info(f"Triplet loss weight: {self.config.training.triplet_loss_weight}")
        logging.info(f"Triples loss margin: {self.config.training.triplet_loss_margin}")

    def load_from_checkpoint_lora(self, path):
        checkpoint = torch.load(path)
        if self.config.training.use_text_proj:
            self.text_proj.load_state_dict(checkpoint['text_proj'])
        if self.config.training.use_image_proj:
            self.image_proj.load_state_dict(checkpoint['image_proj'])
        self.logit_scale.load_state_dict(checkpoint['logit_scale']) #module.logit_scale = checkpoint['logit_scale']
        if self.config.training.use_openclip_optimizer_scheduler == False:
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

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if self.config.training.use_text_proj:
            self.text_proj.load_state_dict(checkpoint['text_proj'])
        if self.config.training.use_image_proj:
            self.image_proj.load_state_dict(checkpoint['image_proj'])
        self.logit_scale.load_state_dict(checkpoint['logit_scale']) #module.logit_scale = checkpoint['logit_scale']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.config.training.use_openclip_optimizer_scheduler == False:
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
    
    def load_from_online(self, model_name):
        logging.info("Loading model: {}".format(model_name))

        checkpoint = torch.load(hf_hub_download(repo_id=model_name, filename="model.pt"))
        model_dict = OrderedDict()
        pattern = re.compile('module.')
        for k,v in checkpoint['state_dict'].items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
        self.model.load_state_dict(checkpoint['state_dict'])

        if self.config.training.use_text_proj:
            self.text_proj.load_state_dict(checkpoint['text_proj'])
        if self.config.training.use_image_proj:
            self.image_proj.load_state_dict(checkpoint['image_proj'])
        self.logit_scale.load_state_dict(checkpoint['logit_scale']) #module.logit_scale = checkpoint['logit_scale']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.config.training.use_openclip_optimizer_scheduler == False:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_img_contras_acc = checkpoint['best_img_contras_acc']
        self.best_text_contras_acc = checkpoint['best_text_contras_acc']
        self.best_modelnet40_overall_acc = checkpoint['best_modelnet40_overall_acc']
        self.best_modelnet40_class_acc = checkpoint['best_modelnet40_class_acc']
        self.best_lvis_acc = checkpoint['best_lvis_acc']

        logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))
        logging.info("----Best img contras acc: {}".format(self.best_img_contras_acc))
        logging.info("----Best text contras acc: {}".format(self.best_text_contras_acc))
        logging.info("----Best modelnet40 overall acc: {}".format(self.best_modelnet40_overall_acc))
        logging.info("----Best modelnet40 class acc: {}".format(self.best_modelnet40_class_acc))
        logging.info("----Best lvis acc: {}".format(self.best_lvis_acc))

    def contras_loss(self, feat1, feat2, logit_scale=1, mask = None):
        # feat1 batch_size x 1280   feat2 batch_size x 1280  
        if self.config.ngpu > 1:
            feat1 = F.normalize(feat1, dim=1)
            feat2 = F.normalize(feat2, dim=1)
            all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
            all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
            logits = logit_scale * all_feat1 @ all_feat2.T
        else:
            logits = logit_scale * F.normalize(feat1, dim=1) @ F.normalize(feat2, dim=1).T
        if mask is not None:
            logits = logits * mask
        labels = torch.arange(logits.shape[0]).to(self.config.device)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss, accuracy
    
    def triplet_loss_old(self, pred_feat, txt_feat, img_feat, logit_scale=1, margin=0.4):
        # Calculating the nearest neighbors from batch features to lvis test features 
        _, txt_nbr_ind = self.text_feat_nbrs.kneighbors(txt_feat)
        _, img_nbr_ind = self.img_feat_nbrs.kneighbors(img_feat)

        # This matrix is created to store the nearest neighbor features for each type of feature
        txt_feat_nbr_mat = torch.from_numpy(self.lvis_eval_text_feat[txt_nbr_ind[:, 0]])
        img_feat_nbr_mat = torch.from_numpy(self.lvis_eval_img_feat[img_nbr_ind[:, 0]])
        
        pos_txt_logits = logit_scale * F.normalize(pred_feat, dim=1) @ F.normalize(txt_feat, dim=1).T
        neg_txt_logits = logit_scale * F.normalize(pred_feat, dim=1) @ F.normalize(txt_feat_nbr_mat, dim=1).T
        pos_img_logits = logit_scale * F.normalize(pred_feat, dim=1) @ F.normalize(img_feat, dim=1).T
        neg_img_logits = logit_scale * F.normalize(pred_feat, dim=1) @ F.normalize(img_feat_nbr_mat, dim=1).T

        # Calculating the triplet loss
        txt_triplet_loss = torch.clamp(neg_txt_logits - pos_txt_logits + margin, 0, None) 
        img_triplet_loss = torch.clamp(neg_img_logits - pos_img_logits + margin, 0, None)
        triplet_loss = txt_triplet_loss.mean() + img_triplet_loss.mean()
        return triplet_loss
    
    def triplet_loss(self, pred_feat, txt_feat, img_feat, logit_scale=1, margin=0.4):
        # Calculating the nearest neighbors from batch features to lvis test features 
        _, txt_nbr_ind = self.text_feat_nbrs.kneighbors(txt_feat)
        _, img_nbr_ind = self.img_feat_nbrs.kneighbors(img_feat)

        # This matrix is created to store the nearest neighbor features for each type of feature
        txt_feat_nbr_mat = torch.from_numpy(self.lvis_eval_text_feat[txt_nbr_ind[:, 0]])
        img_feat_nbr_mat = torch.from_numpy(self.lvis_eval_img_feat[img_nbr_ind[:, 0]])
        
        pos_txt_logits = logit_scale * F.normalize(pred_feat, dim=1) * F.normalize(txt_feat, dim=1)
        neg_txt_logits = logit_scale * F.normalize(pred_feat, dim=1) * F.normalize(txt_feat_nbr_mat, dim=1)
        pos_img_logits = logit_scale * F.normalize(pred_feat, dim=1) * F.normalize(img_feat, dim=1)
        neg_img_logits = logit_scale * F.normalize(pred_feat, dim=1) * F.normalize(img_feat_nbr_mat, dim=1)

        pos_txt_logits = pos_txt_logits.sum(dim=1).view(1, -1)
        neg_txt_logits = neg_txt_logits.sum(dim=1).view(-1, 1)
        pos_img_logits = pos_img_logits.sum(dim=1).view(1, -1)
        neg_img_logits = neg_img_logits.sum(dim=1).view(-1, 1)

        # Calculating the triplet loss
        txt_triplet_loss = torch.clamp(neg_txt_logits - pos_txt_logits + margin, 0, None)
        img_triplet_loss = torch.clamp(neg_img_logits - pos_img_logits + margin, 0, None)
        triplet_loss = txt_triplet_loss.mean() + img_triplet_loss.mean()

        return triplet_loss
    
    def train_one_epoch(self):
        self.model.train()
        if self.config.training.use_text_proj:
            self.text_proj.train()
        if self.config.training.use_image_proj:
            self.image_proj.train()

        text_contras_acc_list = []
        img_contras_acc_list = []
        if self.config.training.use_mask:
            k = self.config.dataset.negative_sample_num
            s = self.config.dataset.train_batch_size
            mask1 = np.eye(k * s).astype(np.bool)
            mask2 = np.kron(np.eye(s), np.ones((k, k))).astype(np.bool)
            mask_other = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)

        for data in tqdm(self.train_loader):
            self.step += 1
            self.optimizer.zero_grad()
            contras_loss = 0
            if not self.config.model.get("use_dense", False):
                pred_feat = self.model(data['xyz'], data['features'], \
                                        device = self.config.device, \
                                        quantization_size = self.config.model.voxel_size)
            else:
                pred_feat = self.model(data['xyz_dense'], data['features_dense'])
            logit_scale = self.logit_scale(None)
            idx = data['has_text_idx']

            if self.config.text_model == "clip":
                text_feat = torch.vstack(data['text_feat']).to(self.config.device)
            elif self.config.text_model == "vico":
                vico_feat = self.vico_feat_dict[data["category"]]
                text_feat = torch.vstack(vico_feat).to(self.config.device)
            img_feat = torch.vstack(data['img_feat']).to(self.config.device) 

            if self.config.training.use_mask:
                img_text_sim = F.normalize(img_feat, dim=-1) @ F.normalize(text_feat, dim=-1).T
                mask = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
                mask = torch.logical_or(mask, mask_other).detach()
            else:
                mask = None

            if len(idx) > 0:
                if self.config.training.use_text_proj:
                    text_feat = self.text_proj(text_feat)

                text_contras_loss, text_contras_acc = self.contras_loss(pred_feat[idx], text_feat, logit_scale=logit_scale, mask=mask)
                contras_loss += text_contras_loss * self.config.training.lambda_text_contras 
                text_contras_acc_list.append(text_contras_acc.item())

            if self.config.training.use_image_proj:
                img_feat = self.image_proj(img_feat)
                
            img_contras_loss, img_contras_acc = self.contras_loss(pred_feat, img_feat, logit_scale=logit_scale, mask=mask)
            contras_loss += img_contras_loss * self.config.training.lambda_img_contras
            loss = contras_loss * self.config.training.contras_loss_weight

            triplet_loss = 0
            if self.config.training.use_triplet_loss:
                triplet_loss = self.triplet_loss(pred_feat.cpu(), text_feat.cpu(), img_feat.cpu(), 
                                                 logit_scale=logit_scale.cpu(), margin=self.config.training.triplet_loss_margin)
                loss += triplet_loss * self.config.training.triplet_loss_weight

            loss.backward()
            self.optimizer.step()
            if self.config.training.use_openclip_optimizer_scheduler:
                self.scheduler(self.step)
            else:
                self.scheduler.step()
            
            img_contras_acc_list.append(img_contras_acc.item())

        if self.rank == 0 and self.step % self.config.training.log_freq == 0:
            try:
                if self.config.wandb_key is not None:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/contras_loss": contras_loss.item() if len(idx) > 0 else 0,
                        "train/text_contras_acc": text_contras_acc.item() if len(idx) > 0 else 0,
                        "train/img_contras_acc": img_contras_acc.item(),
                        "train/triplet_loss": triplet_loss.item(),
                        "train/lr": self.optimizer.param_groups[0]['lr'],
                        "train/epoch": self.epoch,
                        "train/step": self.step,
                        "train/logit_scale": logit_scale,
                        "train/has_text": len(idx),
                        "train/filtered_pair": (mask == False).sum() if mask is not None else 0
                    })
            except:
                print("wandb log error", flush=True)
        if self.rank == 0: 
            logging.info('Train: text_cotras_acc: {0} image_contras_acc: {1}'\
                    .format(np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0,
                            np.mean(img_contras_acc_list)))
            logging.info(f"Text contrastive loss: {text_contras_loss.item() if len(idx) > 0 else 0}")
            logging.info(f"Image contrastive loss: {img_contras_loss.item()}")
            logging.info(f"Triplet loss: {triplet_loss.item()}")

    def save_model(self, name):
        torch_dict = {
            "state_dict": self.model.state_dict(),
            "logit_scale": self.logit_scale.state_dict(),#module.logit_scale,
            "text_proj": self.text_proj.state_dict() if self.config.training.use_text_proj else None,
            "image_proj": self.image_proj.state_dict() if self.config.training.use_image_proj else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.config.training.use_openclip_optimizer_scheduler == False else None,
            "epoch": self.epoch,
            "step": self.step,
            "best_img_contras_acc": self.best_img_contras_acc,
            "best_text_contras_acc": self.best_text_contras_acc,
            "best_modelnet40_overall_acc": self.best_modelnet40_overall_acc,
            "best_modelnet40_class_acc": self.best_modelnet40_class_acc,
            "best_lvis_acc": self.best_lvis_acc,
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
        best_lvis_acc = 0
        for epoch in range(self.epoch, self.config.training.max_epoch):
            self.epoch = epoch
            if self.rank == 0:
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
            
            if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
                overall_acc = self.test_objaverse_lvis()
                if overall_acc > best_lvis_acc:
                    best_lvis_acc = overall_acc
                    self.save_model('best_lvis')
                    logging.info("Best LVIS acc: {}".format(best_lvis_acc))

    def test_modelnet40(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.modelnet40_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(40).to(self.config.device)
        per_cat_count = torch.zeros(40).to(self.config.device)
        category2idx = self.modelnet40_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}
        
        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in self.modelnet40_loader:
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                            device = self.config.device, \
                                            quantization_size = self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)

                for i in range(40):
                    idx = (labels == i)
                    if idx.sum() > 0:
                        per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
                        per_cat_count[i] += idx.sum()
        topk_acc, correct = self.accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1,3,5,))

        overall_acc = per_cat_correct.sum() / per_cat_count.sum()
        per_cat_acc = per_cat_correct / per_cat_count

        if overall_acc > self.best_modelnet40_overall_acc:
            self.best_modelnet40_overall_acc = overall_acc
            self.save_model('best_modelnet40_overall')
        if per_cat_acc.mean() > self.best_modelnet40_class_acc:
            self.best_modelnet40_class_acc = per_cat_acc.mean()
            self.save_model('best_modelnet40_class')

        logging.info('Test ModelNet40: overall acc: {0}({1}) class_acc: {2}({3})'.format(overall_acc, self.best_modelnet40_overall_acc, per_cat_acc.mean(), self.best_modelnet40_class_acc))
        logging.info('Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()))
        if self.config.wandb_key is not None:
            wandb.log({"test/epoch": self.epoch,
                    "test/step": self.step,
                    "test/ModelNet40_overall_acc": overall_acc,
                    "test/ModelNet40_class_acc": per_cat_acc.mean(),
                    "test/top3_acc": topk_acc[1],
                    "test/top5_acc": topk_acc[2],})

    def test_objaverse_lvis(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.objaverse_lvis_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(1156).to(self.config.device)
        per_cat_count = torch.zeros(1156).to(self.config.device)
        category2idx = self.objaverse_lvis_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}
        
        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in tqdm(self.objaverse_lvis_loader):
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                            device = self.config.device, \
                                            quantization_size = self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)
                # calculate per class accuracy
                for i in torch.unique(labels):
                    idx = (labels == i)
                    if idx.sum() > 0:
                        per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
                        per_cat_count[i] += idx.sum()
        topk_acc, correct = self.accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1,3,5,))

        overall_acc = per_cat_correct.sum() / per_cat_count.sum()
        per_cat_acc = per_cat_correct / per_cat_count

        if overall_acc > self.best_lvis_acc:
            self.best_lvis_acc = overall_acc
            self.save_model('best_lvis')

        logging.info('Test ObjaverseLVIS: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
        logging.info('Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()))
        if self.config.wandb_key is not None:
            wandb.log({"test_lvis/epoch": self.epoch,
                    "test_lvis/step": self.step,
                    "test_lvis/overall_acc": overall_acc,
                    "test_lvis/class_acc": per_cat_acc.mean(),
                    "test_lvis/top3_acc": topk_acc[1],
                    "test_lvis/top5_acc": topk_acc[2],})
        
        return overall_acc
   
    def test_scanobjectnn(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(15).to(self.config.device)
        per_cat_count = torch.zeros(15).to(self.config.device)
        category2idx = self.scanobjectnn_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}
        
        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in self.scanobjectnn_loader:
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                            device = self.config.device, \
                                            quantization_size = self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)
                # calculate per class accuracy
                for i in range(15):
                    idx = (labels == i)
                    if idx.sum() > 0:
                        per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
                        per_cat_count[i] += idx.sum()

        topk_acc, correct = self.accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1,3,5,))

        overall_acc = per_cat_correct.sum() / per_cat_count.sum()
        per_cat_acc = per_cat_correct / per_cat_count

        logging.info('Test ScanObjectNN: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
        logging.info('Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()))
        if self.config.wandb_key is not None:
            wandb.log({"test_scanobjectnn/epoch": self.epoch,
                    "test_scanobjectnn/step": self.step,
                    "test_scanobjectnn/overall_acc": overall_acc,
                    "test_scanobjectnn/class_acc": per_cat_acc.mean(),
                    "test_scanobjectnn/top3_acc": topk_acc[1],
                    "test_scanobjectnn/top5_acc": topk_acc[2],})
