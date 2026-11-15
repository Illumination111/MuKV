"""Unchanged MuKV operators extracted from model/mukv_multigrain_kvcache.py.

Kept separate to avoid importing the LLaVA/FlashAttention backbone for Qwen.
Original repository license applies.
"""
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
logger = logging.getLogger(__name__)

class MuKVMultiGrainKVCache:
    """MuKV 的多粒度 KV-Cache 存储器。"""

    def __init__(self, granularities: List[int], device: str = 'cuda'):
        """
        Args:
            granularities: 粒度列表，如[49, 196, 784]
        """
        self.granularities = granularities
        self.device = device

        # 为每个粒度创建独立的KV-Cache存储
        self.kv_stores = {
            gran: {
                'keys': [],           # List[List[(k, v)]] - 所有blocks的KV
                'repr_vectors': [],   # List[torch.Tensor] - 代表向量
                'token_indices': [],  # List[torch.Tensor] - 保留的token位置
                'compression_stats': []  # List[dict] - 压缩统计
            }
            for gran in granularities
        }

    def store_kv_cache(
        self,
        granularity: int,
        kv_layers: List[Tuple[torch.Tensor, torch.Tensor]],
        repr_vector: torch.Tensor,
        token_indices: torch.Tensor,
        compression_info: dict
    ):
        """
        存储压缩后的KV-Cache

        Args:
            granularity: 粒度大小
            kv_layers: List[(k, v)] 所有层的压缩后KV
            repr_vector: (hidden_dim,) 代表向量
            token_indices: (n_kept,) 保留的token位置索引
            compression_info: 压缩统计信息
        """
        # 移到CPU节省GPU内存
        kv_layers_cpu = [(k.detach().cpu(), v.detach().cpu()) for k, v in kv_layers]
        repr_cpu = repr_vector.detach().cpu()
        indices_cpu = token_indices.detach().cpu()

        self.kv_stores[granularity]['keys'].append(kv_layers_cpu)
        self.kv_stores[granularity]['repr_vectors'].append(repr_cpu)
        self.kv_stores[granularity]['token_indices'].append(indices_cpu)
        self.kv_stores[granularity]['compression_stats'].append(compression_info)

    def retrieve_kv_cache(
        self,
        granularity: int,
        query_vector: torch.Tensor,
        topk: int
    ) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        从指定粒度检索最相关的KV-Cache

        Args:
            granularity: 粒度
            query_vector: (hidden_dim,) 查询向量
            topk: 检索数量

        Returns:
            selected_kv_layers: List[List[(k, v)]] 检索到的KV列表
        """
        store = self.kv_stores[granularity]
        if len(store['repr_vectors']) == 0:
            return []

        # 计算相似度
        repr_vectors = torch.stack(store['repr_vectors']).to(self.device)
        query_vector_gpu = query_vector.to(self.device)
        similarities = torch.cosine_similarity(
            query_vector_gpu.unsqueeze(0), repr_vectors, dim=1
        )

        # TopK选择
        actual_topk = min(topk, len(similarities))
        top_indices = similarities.topk(actual_topk).indices.cpu().tolist()

        # 获取对应的KV-Cache并移回GPU
        selected_kv_layers = []
        for i in top_indices:
            kv_layers_cpu = store['keys'][i]
            kv_layers_gpu = [(k.to(self.device), v.to(self.device)) for k, v in kv_layers_cpu]
            selected_kv_layers.append(kv_layers_gpu)

        logger.debug(f"Granularity {granularity}: retrieved {len(selected_kv_layers)} blocks, "
                    f"max_sim={similarities.max().item():.3f}")

        return selected_kv_layers

    def get_stats(self):
        """获取存储统计信息"""
        stats = {}
        for gran in self.granularities:
            n_blocks = len(self.kv_stores[gran]['keys'])
            if n_blocks > 0:
                avg_tokens = np.mean([
                    len(indices) for indices in self.kv_stores[gran]['token_indices']
                ])
                avg_ratio = avg_tokens / gran if gran > 0 else 0
            else:
                avg_tokens = 0
                avg_ratio = 0

            stats[gran] = {
                'n_blocks': n_blocks,
                'avg_tokens_per_block': avg_tokens,
                'avg_keep_ratio': avg_ratio
            }
        return stats

class TokenSelector:
    def _select_tokens_with_dedup(
        self,
        full_kvs: List[Tuple[torch.Tensor, torch.Tensor]],
        importance_scores: torch.Tensor,
        target_k: int,
        similarity_threshold: float
    ) -> torch.Tensor:
        """
        思路3：去冗余选择tokens

        Args:
            full_kvs: 所有层的完整KV
            importance_scores: (seq_len,) token重要性
            target_k: 目标保留数量
            similarity_threshold: 相似度阈值

        Returns:
            final_indices: 最终保留的token索引
        """
        seq_len = importance_scores.shape[0]

        # 先选1.2倍候选（为去重留余地）
        candidate_k = min(int(target_k * 1.2), seq_len)
        candidate_indices = importance_scores.topk(candidate_k).indices

        # 🆕 确保类型和内存连续性
        candidate_indices = candidate_indices.long().contiguous()

        if not self.dedup_enabled or candidate_k <= target_k:
            return candidate_indices[:target_k].contiguous()

        # 提取最后一层的K向量
        last_layer_k = full_kvs[-1][0]  # (batch, n_heads, seq_len, dim)
        last_layer_k = last_layer_k.mean(dim=1).squeeze(0)  # (seq_len, dim)
        candidate_k_vectors = last_layer_k[candidate_indices]  # (candidate_k, dim)

        # 计算相似度矩阵（手动计算，避免torch.cosine_similarity的维度问题）
        # 归一化向量
        candidate_k_norm = candidate_k_vectors / candidate_k_vectors.norm(dim=1, keepdim=True)
        # 矩阵乘法得到余弦相似度
        sim_matrix = torch.mm(candidate_k_norm, candidate_k_norm.t())  # (candidate_k, candidate_k)

        # 去除冗余tokens
        keep_mask = torch.ones(candidate_k, dtype=torch.bool, device=sim_matrix.device)

        for i in range(candidate_k):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, candidate_k):
                if sim_matrix[i, j] > similarity_threshold:
                    # 保留importance更高的
                    idx_i = candidate_indices[i]
                    idx_j = candidate_indices[j]
                    if importance_scores[idx_i] < importance_scores[idx_j]:
                        keep_mask[i] = False
                        break
                    else:
                        keep_mask[j] = False

        # 最终保留的indices
        kept_indices = candidate_indices[keep_mask]

        # 确保数量不超过target_k
        if len(kept_indices) > target_k:
            # 从kept中再选top-k（按importance）
            kept_importance = importance_scores[kept_indices]
            final_positions = kept_importance.topk(target_k).indices
            final_indices = kept_indices[final_positions]
        else:
            final_indices = kept_indices

        return final_indices
