import random
import dgl
import torch
import torch.nn.functional as F
from .attention_diffusion import GATNet


class DADGNN(torch.nn.Module):
    """
    Diffusion attention GNN on a per-news token graph (n-gram edges).

    Intended role (design / 论文表述):
    - **共性特征**：多条路径上的扩散聚合把全句信息汇到各 token，得到与上下文一致、
      跨位置稳定一致的表示，相当于在句内强化「整篇新闻共有」的语义基底。
    - **簇特征**：n-gram 邻域 + 注意力边权把局部词组/短语形成紧密子结构（簇），
      GAT 迭代相当于在句内发现若干语义簇并传播；再经残差接到主干，供后续 MoE 使用。

    注：这里是 **单条新闻内部的 token 图**；跨新闻、跨话题的簇需依赖 batch 训练与下游门控/分类
    间接体现，而非在本模块里显式建跨文档图。
    """

    def __init__(
        self,
        emb_dim,
        num_hidden,
        num_layers,
        num_heads,
        k,
        alpha,
        n_gram,
        drop_out,
        max_length=350,
        merge='mean',
    ):
        super(DADGNN, self).__init__()
        self.ngram = n_gram
        self.max_length = max_length
        self.emb_dim = emb_dim
        self.gatnet = GATNet(
            emb_dim, emb_dim, num_hidden, num_layers, k, alpha, num_heads, merge=merge
        )
        self.dropout = torch.nn.Dropout(p=drop_out)

    def add_seq_edges(self, doc_ids: list, old_to_new: dict):
        edges = []
        for index, src_word_old in enumerate(doc_ids):
            src = old_to_new[src_word_old]
            for i in range(max(0, index - self.ngram), min(index + self.ngram + 1, len(doc_ids))):
                dst_word_old = doc_ids[i]
                dst = old_to_new[dst_word_old]
                edges.append([src, dst])
        return edges

    def seq_to_graph(self, feature: torch.Tensor) -> dgl.DGLGraph:
        device = feature.device
        doc_ids = list(range(0, feature.shape[0]))
        random.shuffle(doc_ids)
        local_vocab = set(doc_ids)
        old_to_new = dict(zip(local_vocab, range(len(local_vocab))))
        num_nodes = len(local_vocab)
        k_feat = torch.zeros(num_nodes, feature.shape[1], device=device, dtype=feature.dtype)
        for old_idx, new_idx in old_to_new.items():
            k_feat[new_idx] = feature[old_idx]
        seq_edges = self.add_seq_edges(doc_ids, old_to_new)
        srcs, dsts = zip(*seq_edges)
        g = dgl.graph((torch.tensor(srcs, device=device), torch.tensor(dsts, device=device)), num_nodes=num_nodes)
        g = g.to(device)
        g.ndata['k'] = k_feat
        return g

    def forward(self, features, mask):
        """
        features: (batch, seq_len, emb_dim)
        mask: (batch, seq_len) 1 for valid tokens
        """
        bsz, max_len, dim = features.shape
        assert dim == self.emb_dim
        out = torch.zeros_like(features)
        sub_graphs = []
        lengths = []
        for i in range(bsz):
            length = int(mask[i].sum().item())
            length = max(min(length, max_len), 1)
            lengths.append(length)
            fi = features[i, :length]
            sub_graphs.append(self.seq_to_graph(fi))
        batch_graph = dgl.batch(sub_graphs)
        batch_f = self.dropout(batch_graph.ndata['k'])
        h1 = self.gatnet(batch_graph, batch_f)
        batch_graph.ndata['h_out'] = h1
        graphs = dgl.unbatch(batch_graph)
        for i, g in enumerate(graphs):
            ln = lengths[i]
            out[i, :ln] = g.ndata['h_out']
        return out
