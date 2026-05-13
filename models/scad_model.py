import os
import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from utils.utils import data2gpu, Averager, metrics, Recorder
from .orthographic import ortho_algorithm
from .sc_attention import ContrastiveLoss, default_contrastive_opt
from .attention_diffusion import AttentionDiffusion
from .domain_subspace import ContinuousDomainSubspace


class MultiDomainFENDModel(torch.nn.Module):
    def __init__(
        self,
        emb_dim,
        mlp_dims,
        bert,
        dropout,
        emb_type,
        use_dadgnn=True,
        dadgnn_residual=0.5,
        use_ortho=True,
        dadgnn_num_layers=2,
        dadgnn_num_heads=4,
        dadgnn_k=3,
        dadgnn_alpha=0.2,
        dadgnn_ngram=3,
        use_grl=True,
        grl_alpha=1.0,
        use_scad_paper=True,
        tau_diffusion=0.65,
        gamma_diffusion=0.15,
        T_diffusion=5,
        subspace_mix_tau=0.1,
    ):
        super(MultiDomainFENDModel, self).__init__()
        self.domain_num = 9
        self.num_expert = 5
        self.fea_size = 256
        self.d_s = 256
        self.d_u = 256
        self.emb_type = emb_type
        self.use_dadgnn = use_dadgnn
        self.dadgnn_residual = dadgnn_residual
        self.use_ortho = use_ortho
        self.use_grl = use_grl
        self.use_scad_paper = use_scad_paper

        if emb_type == 'bert':
            self.bert = BertModel.from_pretrained(bert).requires_grad_(False)

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        expert = []
        for i in range(self.num_expert):
            expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.expert = nn.ModuleList(expert)

        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 2, mlp_dims[-1]),
            nn.ReLU(),
            nn.Linear(mlp_dims[-1], self.num_expert),
            nn.Softmax(dim=1),
        )

        self.attention = MaskAttention(emb_dim)

        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim)
        self.specific_extractor = SelfAttentionFeatureExtract(
            multi_head_num=1, input_size=emb_dim, output_size=self.fea_size
        )

        if self.use_scad_paper:
            self.attn_diffusion = AttentionDiffusion(
                emb_dim, self.d_s, tau_s=tau_diffusion, gamma=gamma_diffusion, T=T_diffusion
            )
            self.domain_sub = ContinuousDomainSubspace(
                emb_dim, self.d_u, self.domain_num, mix_temperature=subspace_mix_tau
            )
            self.alpha_gate = nn.Sequential(
                nn.Linear(self.d_s + self.d_u, mlp_dims[-1]),
                nn.ReLU(),
                nn.Linear(mlp_dims[-1], 1),
                nn.Sigmoid(),
            )
            self.grl_zs = GradientReversalLayer(alpha=grl_alpha)
            hid = mlp_dims[-1]
            self.domain_disc_adv = nn.Sequential(
                nn.Linear(self.d_s, hid),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hid, self.domain_num),
            )
            self.domain_disc_u = nn.Sequential(
                nn.Linear(self.d_u, hid),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hid, self.domain_num),
            )
            clf_in = 320 + self.d_s
        else:
            self.grl_zs = None
            self.domain_disc_adv = None
            self.domain_disc_u = None
            clf_in = 320

        self.classifier = MLP(clf_in, mlp_dims, dropout)

        if self.use_grl and not self.use_scad_paper:
            self.grl = GradientReversalLayer(alpha=grl_alpha)
            self.domain_discriminator = nn.Sequential(
                nn.Linear(emb_dim, mlp_dims[-1]),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(mlp_dims[-1], self.domain_num),
            )
        else:
            self.grl = None
            self.domain_discriminator = None

        if self.use_dadgnn:
            from .dadgnn import DADGNN

            self.dadgnn = DADGNN(
                emb_dim=emb_dim,
                num_hidden=emb_dim,
                num_layers=dadgnn_num_layers,
                num_heads=dadgnn_num_heads,
                k=dadgnn_k,
                alpha=dadgnn_alpha,
                n_gram=dadgnn_ngram,
                drop_out=dropout,
                merge='mean',
            )

    def _h_cls(self, init_feature, masks):
        if self.emb_type == 'bert':
            return init_feature[:, 0, :].contiguous()
        m = masks.float().unsqueeze(-1)
        num = (init_feature * m).sum(dim=1)
        den = m.sum(dim=1).clamp(min=1.0)
        return (num / den).contiguous()

    def forward(self, **kwargs):
        return_aux = kwargs.pop('return_aux', False)
        return_domain_logits = kwargs.pop('return_domain_logits', False)
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        category = kwargs['category']

        if self.emb_type == 'bert':
            init_feature = self.bert(inputs, attention_mask=masks)[0]
        elif self.emb_type == 'w2v':
            init_feature = inputs

        h_cls_in = self._h_cls(init_feature, masks)

        view_a = init_feature.clone()
        if self.use_dadgnn:
            enh = self.dadgnn(init_feature, masks)
            init_feature = init_feature + self.dadgnn_residual * enh
        view_b = init_feature.clone()

        pooled, _ = self.attention(init_feature, masks)
        if torch.is_tensor(category):
            idxs = category.view(-1, 1).long().clamp(0, self.domain_num - 1)
        else:
            idxs = torch.tensor(list(category), device=init_feature.device, dtype=torch.long).view(-1, 1).clamp(
                0, self.domain_num - 1
            )
        domain_embedding = self.domain_embedder(idxs).squeeze(1)

        gate_feature = ortho_algorithm(pooled, domain_embedding) if self.use_ortho else pooled

        gate_input = torch.cat([domain_embedding, gate_feature], dim=-1)
        gate_value = self.gate(gate_input)

        shared_feature = 0
        for i in range(self.num_expert):
            tmp_feature = self.expert[i](init_feature)
            shared_feature += tmp_feature * gate_value[:, i].unsqueeze(1)

        if self.use_scad_paper:
            zs = self.attn_diffusion(init_feature, masks)
            zu = self.domain_sub(h_cls_in)
            alpha = self.alpha_gate(torch.cat([zs, zu], dim=-1))
            z_fused = alpha * zs + (1.0 - alpha) * zu
            label_pred = self.classifier(torch.cat([z_fused, shared_feature], dim=-1))
        else:
            label_pred = self.classifier(shared_feature)

        out = torch.sigmoid(label_pred.squeeze(1))

        extra = {}
        if return_aux:
            extra['view_a'] = view_a
            extra['view_b'] = view_b
            extra['cap_lens'] = masks.sum(dim=1).long().clamp(min=1).tolist()
        if return_domain_logits and self.use_grl and self.use_scad_paper:
            extra['domain_logits_adv'] = self.domain_disc_adv(self.grl_zs(zs))
        if return_domain_logits and self.use_scad_paper:
            extra['domain_logits_u'] = self.domain_disc_u(zu)
            extra['zs'] = zs
            extra['zu'] = zu
            extra['proj_penalty'] = self.domain_sub.projection_orth_penalty()
        elif return_domain_logits and self.use_grl and self.domain_discriminator is not None:
            extra['domain_logits'] = self.domain_discriminator(self.grl(pooled))

        if extra:
            return out, extra
        return out


def orthogonality_penalty(zs, zu):
    """L_orth: Frobenius norm of z_s^T z_u over batch (paper Eq. 11 style)."""
    m = torch.matmul(zs.transpose(0, 1), zu)
    ds, du = zs.size(-1), zu.size(-1)
    return (m ** 2).sum() / (ds * du)


class Trainer():
    def __init__(
        self,
        emb_dim,
        mlp_dims,
        bert,
        use_cuda,
        lr,
        dropout,
        train_loader,
        val_loader,
        test_loader,
        category_dict,
        weight_decay,
        save_param_dir,
        emb_type='bert',
        loss_weight=[1, 0.006, 0.009, 5e-5],
        early_stop=5,
        epoches=100,
        use_dadgnn=True,
        dadgnn_residual=0.5,
        use_ortho=True,
        use_sc_loss=False,
        sc_loss_weight=0.05,
        dadgnn_num_layers=2,
        dadgnn_num_heads=4,
        dadgnn_k=3,
        dadgnn_alpha=0.2,
        dadgnn_ngram=3,
        use_grl=True,
        grl_alpha=1.0,
        adv_loss_weight=0.1,
        use_scad_paper=True,
        dom_loss_weight=0.5,
        orth_loss_weight=0.01,
        proj_loss_weight=0.1,
        tau_diffusion=0.65,
        gamma_diffusion=0.15,
        T_diffusion=5,
        subspace_mix_tau=0.1,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.loss_weight = loss_weight
        self.use_cuda = use_cuda

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.bert = bert
        self.dropout = dropout
        self.emb_type = emb_type
        self.use_dadgnn = use_dadgnn
        self.dadgnn_residual = dadgnn_residual
        self.use_ortho = use_ortho
        self.use_sc_loss = use_sc_loss
        self.sc_loss_weight = sc_loss_weight
        self.dadgnn_num_layers = dadgnn_num_layers
        self.dadgnn_num_heads = dadgnn_num_heads
        self.dadgnn_k = dadgnn_k
        self.dadgnn_alpha = dadgnn_alpha
        self.dadgnn_ngram = dadgnn_ngram
        self.use_grl = use_grl
        self.grl_alpha = grl_alpha
        self.adv_loss_weight = adv_loss_weight
        self.use_scad_paper = use_scad_paper
        self.dom_loss_weight = dom_loss_weight
        self.orth_loss_weight = orth_loss_weight
        self.proj_loss_weight = proj_loss_weight
        self.tau_diffusion = tau_diffusion
        self.gamma_diffusion = gamma_diffusion
        self.T_diffusion = T_diffusion
        self.subspace_mix_tau = subspace_mix_tau

        if not os.path.exists(save_param_dir):
            os.makedirs(save_param_dir, exist_ok=True)
        self.save_param_dir = save_param_dir

        opt = default_contrastive_opt()
        self.sc_loss_fn = ContrastiveLoss(opt)

    def train(self, logger=None):
        if logger:
            logger.info('start training......')
        if self.use_cuda and not torch.cuda.is_available():
            print('Warning: use_cuda is True but CUDA is not available; using CPU.')
            self.use_cuda = False
        self.model = MultiDomainFENDModel(
            self.emb_dim,
            self.mlp_dims,
            self.bert,
            self.dropout,
            self.emb_type,
            use_dadgnn=self.use_dadgnn,
            dadgnn_residual=self.dadgnn_residual,
            use_ortho=self.use_ortho,
            dadgnn_num_layers=self.dadgnn_num_layers,
            dadgnn_num_heads=self.dadgnn_num_heads,
            dadgnn_k=self.dadgnn_k,
            dadgnn_alpha=self.dadgnn_alpha,
            dadgnn_ngram=self.dadgnn_ngram,
            use_grl=self.use_grl,
            grl_alpha=self.grl_alpha,
            use_scad_paper=self.use_scad_paper,
            tau_diffusion=self.tau_diffusion,
            gamma_diffusion=self.gamma_diffusion,
            T_diffusion=self.T_diffusion,
            subspace_mix_tau=self.subspace_mix_tau,
        )
        if self.use_cuda:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                label = batch_data['label']
                optimizer.zero_grad()

                batch_data['return_aux'] = bool(self.use_sc_loss and self.use_dadgnn)
                batch_data['return_domain_logits'] = bool(self.use_grl or self.use_scad_paper)

                out = self.model(**batch_data)
                if isinstance(out, tuple):
                    label_pred, extra = out
                else:
                    label_pred = out
                    extra = {}

                loss_det = loss_fn(label_pred, label.float())
                dev = label_pred.device
                dt = label_pred.dtype
                loss_adv = torch.zeros((), device=dev, dtype=dt)
                loss_dom = torch.zeros((), device=dev, dtype=dt)
                loss_orth = torch.zeros((), device=dev, dtype=dt)
                loss_proj = torch.zeros((), device=dev, dtype=dt)
                dom_target = batch_data['category'].long().view(-1).clamp(0, self.model.domain_num - 1)

                if self.use_scad_paper and 'domain_logits_u' in extra:
                    loss_dom = F.cross_entropy(extra['domain_logits_u'], dom_target)
                    loss_orth = orthogonality_penalty(extra['zs'], extra['zu'])
                    loss_proj = extra['proj_penalty']
                if self.use_grl and self.use_scad_paper and 'domain_logits_adv' in extra:
                    loss_adv = F.cross_entropy(extra['domain_logits_adv'], dom_target)
                elif self.use_grl and not self.use_scad_paper and 'domain_logits' in extra:
                    loss_adv = F.cross_entropy(extra['domain_logits'], dom_target)

                loss_sc = torch.zeros((), device=dev, dtype=dt)
                if self.use_sc_loss and self.use_dadgnn and 'view_a' in extra:
                    loss_sc = self.sc_loss_fn(
                        extra['view_a'], extra['view_b'], extra['cap_lens']
                    )

                loss = (
                    loss_det
                    + self.adv_loss_weight * loss_adv
                    + self.dom_loss_weight * loss_dom
                    + self.orth_loss_weight * loss_orth
                    + self.proj_loss_weight * loss_proj
                    + self.sc_loss_weight * loss_sc
                )

                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                avg_loss.add(loss.item())

            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            results = self.test(self.val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_param_dir, 'parameter_mdfend.pkl'),
                )
            elif mark == 'esc':
                break
            else:
                continue
        ckpt_path = os.path.join(self.save_param_dir, 'parameter_mdfend.pkl')
        map_loc = torch.device('cuda' if self.use_cuda and torch.cuda.is_available() else 'cpu')
        if os.path.isfile(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=map_loc))
        else:
            print(
                'Warning: no checkpoint at %s (validation never improved or no save). '
                'Using last epoch weights for test.' % ckpt_path
            )
        results = self.test(self.test_loader)
        print(results)
        return results, os.path.join(self.save_param_dir, 'parameter_mdfend.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                pred_raw = self.model(**batch_data)
                if isinstance(pred_raw, tuple):
                    batch_label_pred = pred_raw[0]
                else:
                    batch_label_pred = pred_raw

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        return metrics(label, pred, category, self.category_dict)
