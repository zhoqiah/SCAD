#!/usr/bin/python
# -*- coding: utf-8 -*-

# from gensim.models.keyedvectors import Vocab
# from transformers.file_utils import CONFIG_NAME
from utils.dataloader import w2v_data
import torch
import tqdm
import pickle
import logging
import os
import time
import json
from copy import deepcopy

from utils.utils import Averager
from utils.dataloader import bert_data
from models.scad_model import Trainer as MDFENDTrainer

class Run():
    def __init__(self,
                 config
                 ):
        self.configinfo = config

        self.use_cuda = config['use_cuda']
        self.model_name = config['model_name']
        self.lr = config['lr']
        self.batchsize = config['batchsize']
        self.emb_type = config['emb_type']
        self.emb_dim = config['emb_dim']
        self.max_len = config['max_len']
        self.num_workers = config['num_workers']
        self.vocab_file = config['vocab_file']
        self.early_stop = config['early_stop']
        self.bert = config['bert']
        self.root_path = config['root_path']
        self.mlp_dims = config['model']['mlp']['dims']
        self.dropout = config['model']['mlp']['dropout']
        self.seed = config['seed']
        self.weight_decay = config['weight_decay']
        self.epoch = config['epoch']
        self.save_param_dir = config['save_param_dir']
        self.use_dadgnn = config.get('use_dadgnn', True)
        self.use_ortho = config.get('use_ortho', True)
        self.use_sc_loss = config.get('use_sc_loss', False)
        self.dadgnn_residual = config.get('dadgnn_residual', 0.5)
        self.sc_loss_weight = config.get('sc_loss_weight', 0.05)
        self.dadgnn_num_layers = config.get('dadgnn_num_layers', 2)
        self.dadgnn_num_heads = config.get('dadgnn_num_heads', 4)
        self.dadgnn_k = config.get('dadgnn_k', 3)
        self.dadgnn_alpha = config.get('dadgnn_alpha', 0.2)
        self.dadgnn_ngram = config.get('dadgnn_ngram', 3)
        self.use_grl = config.get('use_grl', True)
        self.grl_alpha = config.get('grl_alpha', 1.0)
        self.adv_loss_weight = config.get('adv_loss_weight', 0.1)
        self.use_scad_paper = config.get('use_scad_paper', True)
        self.dom_loss_weight = config.get('dom_loss_weight', 0.5)
        self.orth_loss_weight = config.get('orth_loss_weight', 0.01)
        self.proj_loss_weight = config.get('proj_loss_weight', 0.1)
        self.tau_diffusion = config.get('tau_diffusion', 0.65)
        self.gamma_diffusion = config.get('gamma_diffusion', 0.15)
        self.T_diffusion = config.get('T_diffusion', 5)
        self.subspace_mix_tau = config.get('subspace_mix_tau', 0.1)

        self.train_path = self.root_path + 'train.pkl'
        self.val_path = self.root_path + 'val.pkl'
        self.test_path = self.root_path + 'test.pkl'

        self.category_dict = {
            "科技": 0,  
            "军事": 1,  
            "教育考试": 2,  
            "灾难事故": 3,  
            "政治": 4,  
            "医药健康": 5,  
            "财经商业": 6,  
            "文体娱乐": 7,  
            "社会生活": 8
        }

    def get_dataloader(self):
        if self.emb_type == 'bert':
            loader = bert_data(max_len = self.max_len, batch_size = self.batchsize, vocab_file = self.vocab_file,
                        category_dict = self.category_dict, num_workers=self.num_workers)
        elif self.emb_type == 'w2v':
            loader = w2v_data(max_len=self.max_len, vocab_file=self.vocab_file, emb_dim = self.emb_dim,
                    batch_size=self.batchsize, category_dict=self.category_dict, num_workers= self.num_workers)
        train_loader = loader.load_data(self.train_path, True)
        val_loader = loader.load_data(self.val_path, False)
        test_loader = loader.load_data(self.test_path, False)
        return train_loader, val_loader, test_loader
    
    def config2dict(self):
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    def main(self):
        train_loader, val_loader, test_loader = self.get_dataloader()
        if self.model_name == 'mdfend':
            trainer = MDFENDTrainer(
                emb_dim=self.emb_dim,
                mlp_dims=self.mlp_dims,
                bert=self.bert,
                emb_type=self.emb_type,
                use_cuda=self.use_cuda,
                lr=self.lr,
                train_loader=train_loader,
                dropout=self.dropout,
                weight_decay=self.weight_decay,
                val_loader=val_loader,
                test_loader=test_loader,
                category_dict=self.category_dict,
                early_stop=self.early_stop,
                epoches=self.epoch,
                save_param_dir=os.path.join(self.save_param_dir, self.model_name),
                use_dadgnn=self.use_dadgnn,
                use_ortho=self.use_ortho,
                use_sc_loss=self.use_sc_loss,
                dadgnn_residual=self.dadgnn_residual,
                sc_loss_weight=self.sc_loss_weight,
                dadgnn_num_layers=self.dadgnn_num_layers,
                dadgnn_num_heads=self.dadgnn_num_heads,
                dadgnn_k=self.dadgnn_k,
                dadgnn_alpha=self.dadgnn_alpha,
                dadgnn_ngram=self.dadgnn_ngram,
                use_grl=self.use_grl,
                grl_alpha=self.grl_alpha,
                adv_loss_weight=self.adv_loss_weight,
                use_scad_paper=self.use_scad_paper,
                dom_loss_weight=self.dom_loss_weight,
                orth_loss_weight=self.orth_loss_weight,
                proj_loss_weight=self.proj_loss_weight,
                tau_diffusion=self.tau_diffusion,
                gamma_diffusion=self.gamma_diffusion,
                T_diffusion=self.T_diffusion,
                subspace_mix_tau=self.subspace_mix_tau,
            )
            trainer.train()
        else:
            raise ValueError('Unknown model_name: %s' % self.model_name)
