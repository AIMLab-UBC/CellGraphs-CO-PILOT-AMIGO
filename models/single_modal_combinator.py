import torch
import torch.nn

import models.utils


class AverageCombinator(torch.nn.Module):

    def __init__(self):
        super(AverageCombinator, self).__init__()

    def forward(self, embeddings, study_indexes):
        final_batch_size = len(torch.unique(study_indexes))
        output = models.utils.aggregate_with_index(feature=embeddings, index=study_indexes,
                                                   batch_size=final_batch_size)
        return output


class MaxCombinator(torch.nn.Module):

    def __init__(self):
        super(MaxCombinator, self).__init__()

    def forward(self, embeddings, study_indexes):
        final_batch_size = len(torch.unique(study_indexes))
        study_indexes = study_indexes.type(torch.LongTensor).to(embeddings.device)

        # convert to batch-based
        fixed_embedding = torch.zeros((final_batch_size, embeddings.size(1))).to(embeddings.device)
        for id in range(final_batch_size):
            valid_index = study_indexes == id
            fixed_embedding[id] = torch.max(embeddings[valid_index], dim=0, keepdim=False)[0]

        return fixed_embedding


class Linear(AverageCombinator):

    def __init__(self, dim_features):
        super(Linear, self).__init__()
        self.linear = torch.nn.Linear(dim_features, dim_features)

    def forward(self, embeddings, study_indexes):
        embeddings = self.linear(embeddings)
        return super(Linear, self).forward(embeddings, study_indexes)


class AttentionCombinator(torch.nn.Module):

    def __init__(self, dim_features):
        super(AttentionCombinator, self).__init__()
        self._dim_features = dim_features
        self._backbone = torch.nn.Linear(self._dim_features, 1)

    @staticmethod
    def _softmax(attn, index, batch_size):
        attn = torch.exp(attn)
        # Note: skip this part and replace it with mean in the next step
        # attn_sum = torch.zeros_like(attn, dtype=attn.dtype).to(attn.device)
        # attn_sum.scatter_(dim=0, index=index[..., None], src=models.utils.sum_with_index(attn, index, batch_size))
        # attn = attn / attn_sum
        return attn, models.utils.sum_with_index(attn, index, batch_size)

    def forward(self, embedding, study_indexes):
        study_indexes = study_indexes.type(torch.LongTensor).to(embedding.device)
        final_batch_size = len(torch.unique(study_indexes))
        attn = self._backbone(embedding)
        attn, attn_sum = self._softmax(attn, study_indexes, final_batch_size)
        embedding = embedding * attn
        embedding = models.utils.sum_with_index(embedding, study_indexes, final_batch_size) / attn_sum
        return embedding


class TransformerCombinator(torch.nn.Module):

    def __init__(self, dim_features, n_head, pool='mean'):
        super(TransformerCombinator, self).__init__()
        self._pool = pool
        self._dim_features = dim_features
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self._dim_features,
                                                         nhead=n_head,
                                                         dim_feedforward=self._dim_features * 2)
        self._transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, embedding, study_indexes):
        final_batch_size = len(torch.unique(study_indexes))
        study_indexes = study_indexes.type(torch.LongTensor).to(embedding.device)

        # convert to batch-based
        max_n_dim = torch.max(torch.unique(study_indexes, return_counts=True)[1]).item()
        fixed_embedding = torch.zeros((final_batch_size, max_n_dim, self._dim_features)).to(embedding.device)
        fixed_mask = torch.ones((final_batch_size, max_n_dim), dtype=torch.bool).to(embedding.device)
        for id in range(final_batch_size):
            valid_index = study_indexes == id
            fixed_embedding[id, :valid_index.sum()] = embedding[valid_index]
            fixed_mask[id, :valid_index.sum()] = False

        embedding = self._transformer(fixed_embedding, src_key_padding_mask=fixed_mask.t())
        embedding = embedding.mean(dim=1) if self._pool == 'mean' else embedding[:, 0]
        return embedding
