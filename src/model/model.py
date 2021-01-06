from __future__ import print_function
import torch.nn as nn
from transformers.activations import get_activation
from transformers.modeling_electra import ElectraPreTrainedModel, ElectraModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn.functional as F
from torch.autograd import *


class ElectraForSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config, lstm_hidden, label_emb_size, score_emb_size, score_size, num_layer, bilstm_flag):
        super().__init__(config)

        assert score_emb_size == lstm_hidden * 2, "Please set score-embedding-size to twice the lstm-hidden-size"

        # 분류할 라벨의 개수
        self.num_labels = config.num_labels
        # ELECTRA 모델
        self.electra = ElectraModel(config)

        self.n_hidden = lstm_hidden

        self.score_emb = nn.Embedding(score_size, score_emb_size, scale_grad_by_freq=True)
        self.num_layers = num_layer
        self.bidirectional = 2 if bilstm_flag else 1

        self.lstm_first = nn.LSTM(config.hidden_size, lstm_hidden, bidirectional=True, batch_first=True)
        self.lstm_last = nn.LSTM(lstm_hidden * 4, lstm_hidden, num_layers=self.num_layers,
                                 batch_first=True, bidirectional=bilstm_flag)
        self.lstm_score_sequence = nn.LSTM(lstm_hidden * 2 + score_size, lstm_hidden, num_layers=self.num_layers,
                                           batch_first=True, bidirectional=bilstm_flag)

        self.label_attn = multihead_attention(lstm_hidden * 2, num_heads=1, dropout_rate=config.hidden_dropout_prob)
        self.label_attn_last = multihead_attention(lstm_hidden * 2, num_heads=1, dropout_rate=0)

        self.lstm_hidden2senti = nn.Linear(lstm_hidden * 2, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None,
                input_label_seq_tensor=None, input_senti_seq_tensor=None, word_seq_lengths=None):
        discriminator_hidden_states = self.electra(input_ids, attention_mask, token_type_ids)

        # (batch_size, max_length, hidden_size)
        discriminator_hidden_states = discriminator_hidden_states[0]

        self.batch_size = discriminator_hidden_states.shape[0]
        label_embs = self.score_emb(input_label_seq_tensor)

        hidden = None

        lstm_outputs, hidden = self.lstm_first(discriminator_hidden_states, hidden)
        lstm_outputs = self.dropout(lstm_outputs)

        word2score_attention_output = self.label_attn(lstm_outputs, label_embs, label_embs, False)

        # [batch, seq_length, lstm_hiddn * 2 + score_label_emb]
        lstm_outputs = torch.cat([lstm_outputs, word2score_attention_output], dim=-1)

        """
        Last Layer
        """
        lstm_outputs, hidden = self.lstm_last(lstm_outputs, hidden)
        lstm_outputs = self.dropout(lstm_outputs)

        word2score_attention_output = self.label_attn_last(lstm_outputs, label_embs, label_embs, True)

        # [batch, seq_length, lstm_hidden * 2 + score_size]
        lstm_outputs = torch.cat([lstm_outputs, word2score_attention_output], dim=-1)

        lstm_outputs, hidden = self.lstm_score_sequence(lstm_outputs, hidden)

        lstm_last_hidden = hidden[0].transpose(0, 1).contiguous().view(self.batch_size, -1)


        return word2score_attention_output.permute(0, 2, 1), self.lstm_hidden2senti(lstm_last_hidden)


class multihead_attention(nn.Module):

    def __init__(self, num_units, num_heads=1, dropout_rate=0, gpu=True, causality=False):
        '''Applies multihead attention.
        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            causality: Boolean. If true, units that reference the future are masked.
            num_heads: An int. Number of heads.
        '''
        super(multihead_attention, self).__init__()
        self.gpu = gpu
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())

        self.output_dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, queries, keys, values, last_layer=False):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]
        # Linear projections
        Q = self.Q_proj(queries)  # (N, T_q, C)
        K = self.K_proj(keys)  # (N, T_q, C)
        V = self.V_proj(values)  # (N, T_q, C)

        # get dim to concat
        concat_dim = len(Q.shape) - 1

        if concat_dim == 1:
            Q = Q.unsqueeze(dim=1)
            queries = queries.unsqueeze(dim=1)
            concat_dim = 2

        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=concat_dim), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=concat_dim), dim=0)  # (h*N, T_q, C/h)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=concat_dim), dim=0)  # (h*N, T_q, C/h)

        # Multiplication
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Activation
        if not last_layer:
            outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        query_masks = query_masks.reshape([outputs.shape[0], outputs.shape[1], outputs.shape[2]])

        outputs = outputs * query_masks

        # Dropouts
        outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)

        if last_layer:
            return outputs

        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)

        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=concat_dim)  # (N, T_q, C)

        # Residual connection
        # outputs += queries

        return outputs