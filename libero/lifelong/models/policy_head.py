from typing import List

import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F


class DeterministicHead(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024, num_layers=2):

        super().__init__()
        sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]

        if self.action_squash:
            layers += [nn.Tanh()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y


class GMMHead(nn.Module):
    def __init__(
            self,
            # network_kwargs
            input_size,
            output_size,
            hidden_size=1024,
            num_layers=2,
            min_std=0.0001,
            num_modes=5,
            activation="softplus",
            low_eval_noise=False,
            # loss_kwargs
            loss_coef=1.0,
            att_sparsity_loss_coef=1e-2,
            codebook_normalize_loss_coef=1e-2,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.output_size = output_size
        self.min_std = min_std

        if num_layers > 0:
            sizes = [input_size] + [hidden_size] * num_layers
            layers = []
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            layers += [nn.Linear(sizes[-2], sizes[-1])]
            self.share = nn.Sequential(*layers)
        else:
            self.share = nn.Identity()

        self.mean_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logstd_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logits_layer = nn.Linear(hidden_size, num_modes)

        self.low_eval_noise = low_eval_noise
        self.loss_coef = loss_coef
        self.att_sparsity_loss_coef = att_sparsity_loss_coef
        self.codebook_normalize_loss_coef = codebook_normalize_loss_coef

        if activation == "softplus":
            self.actv = F.softplus
        else:
            self.actv = torch.exp

    def forward_fn(self, x):
        # x: (B, input_size)
        share = self.share(x)
        means = self.mean_layer(share).view(-1, self.num_modes, self.output_size)
        means = torch.tanh(means)
        logits = self.logits_layer(share)

        if self.training or not self.low_eval_noise:
            logstds = self.logstd_layer(share).view(
                -1, self.num_modes, self.output_size
            )
            stds = self.actv(logstds) + self.min_std
        else:
            stds = torch.ones_like(means) * 1e-4
        return means, stds, logits

    def forward(self, x):
        if x.ndim == 3:  # 判断维度是否为B，T，E
            means, scales, logits = TensorUtils.time_distributed(x, self.forward_fn)  # 对时序数据的每个时间步应用相同的操作
        elif x.ndim < 3:
            means, scales, logits = self.forward_fn(x)

        compo = D.Normal(loc=means, scale=scales)  # 生成正态分布
        compo = D.Independent(compo, 1)
        mix = D.Categorical(logits=logits)  # 按照logits的概率，在相应的位置进行采样
        gmm = D.MixtureSameFamily(
            mixture_distribution=mix, component_distribution=compo
        )
        return gmm

    def loss_fn(self, gmm, target, reduction="mean"):
        log_probs = gmm.log_prob(target)
        loss = -log_probs
        if reduction == "mean":
            return loss.mean() * self.loss_coef
        elif reduction == "none":
            return loss * self.loss_coef
        elif reduction == "sum":
            return loss.sum() * self.loss_coef
        else:
            raise NotImplementedError

    def att_spa_loss(self, att_list):
        loss = 0
        for key in att_list:
            # loss = loss - att.max(dim=2)[0].sum(dim=1).mean()
            loss = loss - att_list[key].max(dim=2)[0].mean()  # TODO not sure Or dim=-1
        loss = loss / len(att_list)
        return loss * self.att_sparsity_loss_coef

    def codebook_norm_loss(self, codebook):
        sim_mat = torch.matmul(codebook, codebook.transpose(0, 1))
        identity = torch.eye(codebook.shape[0]).to(sim_mat.device)
        diff = sim_mat - identity
        norm = torch.linalg.norm(diff)
        return norm * self.codebook_normalize_loss_coef
