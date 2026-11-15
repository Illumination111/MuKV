"""MuKV multi-granularity CPU archive and query-dependent retrieval for Qwen."""
import torch
from .mukv_ops import MuKVMultiGrainKVCache, TokenSelector
from .qwen_stream import clone_cache, slice_cache, cache_from, concat_cache, pairs


class Policy:
    name = 'MuKV/Qwen2.5-VL'

    def __init__(self, config):
        self.config = config
        self.selector = TokenSelector()
        self.selector.dedup_enabled = True
        self.reset()

    def reset(self):
        self.store = None
        self.pending = {gran: None for gran in self.config.granularities}
        self.selection_trace = {}

    def initialize(self, engine):
        self.store = MuKVMultiGrainKVCache(list(self.config.granularities), str(engine.device))

    def _encode_block(self, engine, features, positions, gran, archive=True):
        prefix = engine.system_cache.get_seq_length()
        output = engine.forward(features, clone_cache(engine.system_cache), positions, attentions=True)
        full = pairs(slice_cache(output.past_key_values, prefix))
        scores = output.attentions[-1].float().mean(0).mean(0).sum(0)[prefix:]
        ratio = self.config.keep_ratios[self.config.granularities.index(gran)]
        target = max(1, int(features.shape[1] * ratio))
        selected = self.selector._select_tokens_with_dedup(full, scores, target, .90).sort().values
        compressed = [(k[:, :, selected].contiguous(), v[:, :, selected].contiguous()) for k, v in full]
        if archive:
            representative = compressed[-1][0].mean(2).flatten()
            self.store.store_kv_cache(gran, compressed, representative, selected,
                                     {'original_tokens': features.shape[1], 'kept_tokens': selected.numel(),
                                      'timestamp_position': int(positions[0, 0, 0])})
        return cache_from(compressed)

    def ingest(self, engine, features, positions, grid):
        for gran in self.config.granularities:
            old = self.pending[gran]
            data = features if old is None else torch.cat((old[0], features), 1)
            pos = positions if old is None else torch.cat((old[1], positions), 2)
            while data.shape[1] >= gran:
                self._encode_block(engine, data[:, :gran], pos[:, :, :gran], gran)
                data, pos = data[:, gran:], pos[:, :, gran:]
            self.pending[gran] = (data.clone(), pos.clone()) if data.shape[1] else None

    def query_cache(self, engine, embeddings):
        length = embeddings.shape[1]
        pos = torch.arange(engine.next_position, engine.next_position + length, device=engine.device).view(1, 1, -1).expand(3, 1, -1)
        output = engine.forward(embeddings, clone_cache(engine.system_cache), pos)
        query = pairs(output.past_key_values)[-1][0][:, :, -1].flatten()
        caches = [clone_cache(engine.system_cache)]
        self.selection_trace = {}
        for gran, topk in zip(self.config.granularities, self.config.granularity_topks):
            retrieved = self.store.retrieve_kv_cache(gran, query, topk)
            self.selection_trace[gran] = len(retrieved)
            caches.extend(cache_from(block) for block in retrieved)
            remainder = self.pending[gran]
            if remainder is not None:
                caches.append(self._encode_block(engine, *remainder, gran, archive=False))
        return concat_cache(caches)

    def state(self):
        return {'cpu_archive': None if self.store is None else self.store.kv_stores,
                'uncommitted_scale_tails': self.pending, 'retrieval_trace': self.selection_trace}
