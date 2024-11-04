# Cell
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim

from ..components.transformer import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from ..components.selfattention import (
    TriangularCausalMask, ProbMask,
    FullAttention, ProbAttention, AttentionLayer
)
from ..components.embed import DataEmbedding
from ...losses.utils import LossFunction

# Cell
class _Informer(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, pred_len, output_attention,
                 enc_in, dec_in, d_model, c_out, embed, freq, dropout,
                 factor, n_heads, d_ff, activation, e_layers,
                 d_layers, distil):
        super(_Informer, self).__init__()
        self.pred_len = pred_len
        self.output_attention = output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq,
                                           dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq,
                                           dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

# Cell
class Informer(pl.LightningModule):
    def __init__(self, seq_len,
                 label_len, pred_len, output_attention,
                 enc_in, dec_in, d_model, c_out, embed, freq, dropout,
                 factor, n_heads, d_ff, activation, e_layers, d_layers, distil,
                 loss_train, loss_valid, loss_hypar, learning_rate,
                 lr_decay, weight_decay, lr_decay_step_size,
                 random_seed):
        super(Informer, self).__init__()

        #------------------------ Model Attributes ------------------------#
        # Architecture parameters
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.d_model = d_model
        self.c_out = c_out
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.factor = factor
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.activation = activation
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.distil = distil

        # Loss functions
        self.loss_train = loss_train
        self.loss_hypar = loss_hypar
        self.loss_valid = loss_valid
        self.loss_fn_train = LossFunction(loss_train,
                                          seasonality=self.loss_hypar)
        self.loss_fn_valid = LossFunction(loss_valid,
                                          seasonality=self.loss_hypar)

        # Regularization and optimization parameters
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.lr_decay_step_size = lr_decay_step_size
        self.random_seed = random_seed

        self.model = _Informer(pred_len, output_attention,
                               enc_in, dec_in, d_model, c_out,
                               embed, freq, dropout,
                               factor, n_heads, d_ff,
                               activation, e_layers,
                               d_layers, distil)

    def forward(self, batch):
        """
        Autoformer needs batch of shape (batch_size, time, series) for y
        and (batch_size, time, exogenous) for x
        and doesnt need X for each time series.
        USE DataLoader from pytorch instead of TimeSeriesLoader.
        """
        Y = batch['Y'].permute(0, 2, 1)
        X = batch['X'][:, 0, :, :].permute(0, 2, 1)
        sample_mask = batch['sample_mask'].permute(0, 2, 1)
        available_mask = batch['available_mask']

        s_begin = 0
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        batch_x = Y[:, s_begin:s_end, :]
        batch_y = Y[:, r_begin:r_end, :]
        batch_x_mark = X[:, s_begin:s_end, :]
        batch_y_mark = X[:, r_begin:r_end, :]
        outsample_mask = sample_mask[:, r_begin:r_end, :]

        dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :])
        dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1)

        if self.output_attention:
            forecast = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            forecast = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        batch_y = batch_y[:, -self.pred_len:, :]
        outsample_mask = outsample_mask[:, -self.pred_len:, :]

        return batch_y, forecast, outsample_mask, Y

    def training_step(self, batch, batch_idx):

        outsample_y, forecast, outsample_mask, Y = self(batch)

        loss = self.loss_fn_train(y=outsample_y,
                                  y_hat=forecast,
                                  mask=outsample_mask,
                                  y_insample=Y)

        self.log('train_loss', loss, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, idx):

        outsample_y, forecast, outsample_mask, Y = self(batch)

        loss = self.loss_fn_valid(y=outsample_y,
                                  y_hat=forecast,
                                  mask=outsample_mask,
                                  y_insample=Y)

        self.log('val_loss', loss, prog_bar=True)

        return loss

    def on_fit_start(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=self.lr_decay_step_size,
                                                 gamma=self.lr_decay)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}