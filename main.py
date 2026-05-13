import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='mdfend')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--max_len', type=int, default=170)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--early_stop', type=int, default=5)
parser.add_argument('--bert_vocab_file', default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt')
parser.add_argument('--root_path', default='./data/weibo21/') 
parser.add_argument('--bert', default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--seed', type=int, default=2026)
parser.add_argument('--gpu', default='0')
parser.add_argument('--bert_emb_dim', type=int, default=768)
parser.add_argument('--w2v_emb_dim', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--emb_type', default='bert')
parser.add_argument('--w2v_vocab_file', default='./pretrained_model/w2v/Tencent_AILab_Chinese_w2v_model.kv')
parser.add_argument('--save_param_dir', default= './param_model')
parser.add_argument('--use_dadgnn', type=int, default=1, help='1: use DADGNN branch on token graphs')
parser.add_argument('--use_ortho', type=int, default=1, help='1: orthographic refinement vs domain embedding')
parser.add_argument('--use_sc_loss', type=int, default=0, help='1: auxiliary contrastive loss (slow; needs DADGNN)')
parser.add_argument('--dadgnn_residual', type=float, default=0.5)
parser.add_argument('--sc_loss_weight', type=float, default=0.05)
parser.add_argument('--dadgnn_num_layers', type=int, default=2)
parser.add_argument('--dadgnn_num_heads', type=int, default=4)
parser.add_argument('--dadgnn_k', type=int, default=3)
parser.add_argument('--dadgnn_alpha', type=float, default=0.2)
parser.add_argument('--dadgnn_ngram', type=int, default=3)
parser.add_argument('--use_grl', type=int, default=1, help='1: GRL + domain adversarial on pooled representation')
parser.add_argument('--grl_alpha', type=float, default=1.0, help='Gradient reversal strength')
parser.add_argument('--adv_loss_weight', type=float, default=0.1, help='Weight for domain adversarial (CE) loss')
parser.add_argument('--use_scad_paper', type=int, default=1, help='1: paper similarity diffusion + subspace + L_dom/L_orth/L_proj')
parser.add_argument('--dom_loss_weight', type=float, default=0.5)
parser.add_argument('--orth_loss_weight', type=float, default=0.01)
parser.add_argument('--proj_loss_weight', type=float, default=0.1)
parser.add_argument('--tau_diffusion', type=float, default=0.65)
parser.add_argument('--gamma_diffusion', type=float, default=0.15)
parser.add_argument('--T_diffusion', type=int, default=5)
parser.add_argument('--subspace_mix_tau', type=float, default=0.1)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from run import Run
import torch
import numpy as np
import random

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if args.emb_type == 'bert':
    emb_dim = args.bert_emb_dim
    vocab_file = args.bert_vocab_file
elif args.emb_type == 'w2v':
    emb_dim = args.w2v_emb_dim
    vocab_file = args.w2v_vocab_file

print('lr: {}; model name: {}; emb_type: {}; batchsize: {}; epoch: {}; gpu: {}; emb_dim: {}'.format(args.lr, args.model_name, args.emb_type,  args.batchsize, args.epoch, args.gpu, emb_dim))


config = {
        'use_cuda': True,
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'early_stop': args.early_stop,
        'num_workers': args.num_workers,
        'vocab_file': vocab_file,
        'emb_type': args.emb_type,
        'bert': args.bert,
        'root_path': args.root_path,
        'weight_decay': 5e-5,
        'model':
            {
            'mlp': {'dims': [384], 'dropout': 0.2}
            },
        'emb_dim': emb_dim,
        'lr': args.lr,
        'epoch': args.epoch,
        'model_name': args.model_name,
        'seed': args.seed,
        'save_param_dir': args.save_param_dir,
        'use_dadgnn': bool(args.use_dadgnn),
        'use_ortho': bool(args.use_ortho),
        'use_sc_loss': bool(args.use_sc_loss),
        'dadgnn_residual': args.dadgnn_residual,
        'sc_loss_weight': args.sc_loss_weight,
        'dadgnn_num_layers': args.dadgnn_num_layers,
        'dadgnn_num_heads': args.dadgnn_num_heads,
        'dadgnn_k': args.dadgnn_k,
        'dadgnn_alpha': args.dadgnn_alpha,
        'dadgnn_ngram': args.dadgnn_ngram,
        'use_grl': bool(args.use_grl),
        'grl_alpha': args.grl_alpha,
        'adv_loss_weight': args.adv_loss_weight,
        'use_scad_paper': bool(args.use_scad_paper),
        'dom_loss_weight': args.dom_loss_weight,
        'orth_loss_weight': args.orth_loss_weight,
        'proj_loss_weight': args.proj_loss_weight,
        'tau_diffusion': args.tau_diffusion,
        'gamma_diffusion': args.gamma_diffusion,
        'T_diffusion': args.T_diffusion,
        'subspace_mix_tau': args.subspace_mix_tau,
        }



if __name__ == '__main__':
    Run(config = config
        ).main()
