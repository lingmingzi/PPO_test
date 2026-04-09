#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase PPO.
峡谷追猎 PPO 神经网络模型。
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features, gain=1.0):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data, gain=gain)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    """Feature-group encoders + shared trunk + Actor/Critic dual heads."""

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_lite"
        self.device = device

        hero_dim, m1_dim, m2_dim, map_dim, legal_dim, progress_dim = Config.FEATURES
        hidden_dim = Config.MODEL_HIDDEN_DIM
        latent_dim = Config.MODEL_LATENT_DIM
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        # Feature-group encoders / 分组特征编码器
        self.hero_encoder = nn.Sequential(
            make_fc_layer(hero_dim, hidden_dim // 2, gain=nn.init.calculate_gain("relu")),
            nn.ReLU(inplace=True),
        )
        self.monster_encoder = nn.Sequential(
            make_fc_layer(m1_dim + m2_dim, hidden_dim // 2, gain=nn.init.calculate_gain("relu")),
            nn.ReLU(inplace=True),
        )
        self.map_encoder = nn.Sequential(
            make_fc_layer(map_dim, hidden_dim // 2, gain=nn.init.calculate_gain("relu")),
            nn.ReLU(inplace=True),
        )
        self.legal_encoder = nn.Sequential(
            make_fc_layer(legal_dim, hidden_dim // 4, gain=nn.init.calculate_gain("relu")),
            nn.ReLU(inplace=True),
        )
        self.progress_encoder = nn.Sequential(
            make_fc_layer(progress_dim, hidden_dim // 4, gain=nn.init.calculate_gain("relu")),
            nn.ReLU(inplace=True),
        )

        fusion_dim = hidden_dim // 2 + hidden_dim // 2 + hidden_dim // 2 + hidden_dim // 4 + hidden_dim // 4

        # Shared trunk / 共享干路
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.backbone = nn.Sequential(
            make_fc_layer(fusion_dim, hidden_dim, gain=nn.init.calculate_gain("relu")),
            nn.ReLU(inplace=True),
            make_fc_layer(hidden_dim, latent_dim, gain=nn.init.calculate_gain("relu")),
            nn.ReLU(inplace=True),
        )

        # Actor head / 策略头
        self.actor_head = nn.Sequential(
            make_fc_layer(latent_dim, latent_dim, gain=nn.init.calculate_gain("relu")),
            nn.ReLU(inplace=True),
            make_fc_layer(latent_dim, action_num, gain=0.01),
        )

        # Critic head / 价值头
        self.critic_head = nn.Sequential(
            make_fc_layer(latent_dim, latent_dim, gain=nn.init.calculate_gain("relu")),
            nn.ReLU(inplace=True),
            make_fc_layer(latent_dim, value_num, gain=1.0),
        )

        self._feature_splits = [hero_dim, m1_dim, m2_dim, map_dim, legal_dim, progress_dim]

    def forward(self, obs, inference=False):
        hero, m1, m2, map_feat, legal_feat, progress = torch.split(obs, self._feature_splits, dim=1)

        hero_latent = self.hero_encoder(hero)
        monster_latent = self.monster_encoder(torch.cat([m1, m2], dim=1))
        map_latent = self.map_encoder(map_feat)
        legal_latent = self.legal_encoder(legal_feat)
        progress_latent = self.progress_encoder(progress)

        fused = torch.cat(
            [hero_latent, monster_latent, map_latent, legal_latent, progress_latent],
            dim=1,
        )
        fused = self.fusion_norm(fused)
        hidden = self.backbone(fused)

        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
