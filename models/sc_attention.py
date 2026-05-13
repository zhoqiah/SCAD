"""Cross-attention scoring and contrastive loss (image–caption style; reused here for two text views)."""
import torch
import torch.nn as nn


def l2norm(X, dim, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    return torch.div(X, norm)


def func_attention(query, context, opt, smooth, eps=1e-8):
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)
    queryT = torch.transpose(query, 1, 2)
    attn = torch.bmm(context, queryT)
    if opt.raw_feature_norm == "softmax":
        attn = attn.view(batch_size * sourceL, queryL)
        attn = torch.softmax(attn, dim=-1)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l2norm":
        attn = l2norm(nn.LeakyReLU(0.1)(attn), 2)
    elif opt.raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)

    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)
    attn = torch.softmax(attn * smooth, dim=-1)
    attn = attn.view(batch_size, queryL, sourceL)
    attnT = torch.transpose(attn, 1, 2).contiguous()
    contextT = torch.transpose(context, 1, 2)
    weightedContext = torch.bmm(contextT, attnT)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    return weightedContext, attnT


def cosine_similarity(x1, x2, dim=2, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim=dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return w12 / (w1 * w2).clamp(min=eps)


def xattn_score_t2i(images, captions, cap_lens, opt):
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        n_word = cap_lens[i]
        if torch.is_tensor(n_word):
            n_word = int(n_word.item())
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)
    return torch.cat(similarities, 1)


def xattn_score_i2t(images, captions, cap_lens, opt):
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        n_word = cap_lens[i]
        if torch.is_tensor(n_word):
            n_word = int(n_word.item())
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)
    return torch.cat(similarities, 1)


class ContrastiveLoss(nn.Module):
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l):
        # Expect im, s as (batch, seq_len, dim). Older retrieval code used (seq, batch, dim)
        # and permuted; that breaks cap_lens (length batch) vs loop range.
        if self.opt.cross_attn == 't2i':
            scores = xattn_score_t2i(im, s, s_l, self.opt)
        elif self.opt.cross_attn == 'i2t':
            scores = xattn_score_i2t(im, s, s_l, self.opt)
        else:
            raise ValueError("unknown cross_attn:", self.opt.cross_attn)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)
        I = torch.eye(scores.size(0), device=scores.device, dtype=torch.bool)
        cost_s = cost_s.masked_fill(I, 0)
        cost_im = cost_im.masked_fill(I, 0)
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


def default_contrastive_opt():
    """Namespace-compatible defaults for ContrastiveLoss."""
    from types import SimpleNamespace
    return SimpleNamespace(
        raw_feature_norm='l2norm',
        lambda_softmax=4.0,
        lambda_lse=4.0,
        agg_func='Mean',
        cross_attn='t2i',
    )
