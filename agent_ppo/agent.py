#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Agent class for Gorge Chase PPO.
峡谷追猎 PPO Agent 主类。
"""

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self._is_cuda_device = bool(
            torch.cuda.is_available() and (self.device is not None) and ("cuda" in str(self.device).lower())
        )

        if self._is_cuda_device:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        if self._is_cuda_device and Config.ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(
                    self.model,
                    mode=Config.TORCH_COMPILE_MODE,
                    dynamic=Config.TORCH_COMPILE_DYNAMIC,
                    fullgraph=Config.TORCH_COMPILE_FULLGRAPH,
                )
                if logger is not None:
                    logger.info("PPO model torch.compile enabled")
            except Exception as e:
                if logger is not None:
                    logger.warning(f"torch.compile disabled due to: {e}")

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=Config.ADAM_EPS,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.predict_count = 0
        self.phase_name = "early"
        self.epsilon_scale = 1.0
        self.phase_flash_explore_prob = float(Config.FLASH_EXPLORE_PROB)
        self.logger = logger
        self.monitor = monitor
        super().__init__(agent_type, device, logger, monitor)

    def set_phase_params(self, phase_name=None, epsilon_scale=None, flash_explore_prob=None):
        if phase_name is not None:
            self.phase_name = str(phase_name)
        if epsilon_scale is not None:
            self.epsilon_scale = float(max(epsilon_scale, 0.1))
        if flash_explore_prob is not None:
            self.phase_flash_explore_prob = float(np.clip(flash_explore_prob, 0.0, 1.0))

    def _get_epsilon(self):
        total = max(int(Config.EPS_STAGE_TOTAL_STEPS), 1)
        p = min(self.predict_count / total, 1.0)

        p1 = float(Config.EPS_STAGE1_RATIO)
        p2 = float(Config.EPS_STAGE2_RATIO)
        eps_max = float(Config.EPSILON_MAX)
        eps1 = float(Config.EPS_STAGE1_VALUE)
        eps2 = float(Config.EPS_STAGE2_VALUE)
        eps_final = float(Config.EPS_FINAL_VALUE)

        if p <= p1:
            t = p / max(p1, 1e-8)
            eps = eps_max + (eps1 - eps_max) * t
        elif p <= p2:
            t = (p - p1) / max(p2 - p1, 1e-8)
            eps = eps1 + (eps2 - eps1) * t
        else:
            t = (p - p2) / max(1.0 - p2, 1e-8)
            eps = eps2 + (eps_final - eps2) * t

        eps = eps * self.epsilon_scale
        return float(np.clip(eps, Config.EPSILON_MIN, Config.EPSILON_MAX))

    def reset(self, env_obs=None):
        """Reset per-episode state.

        每局开始时重置状态。
        """
        self.preprocessor.reset()
        self.last_action = -1

    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData and remain_info.

        将原始观测转换为 ObsData 和 remain_info。
        """
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        remain_info = {"reward": reward}
        return obs_data, remain_info

    def predict(self, list_obs_data):
        """Stochastic inference for training (exploration).

        训练时随机采样动作（探索）。
        """
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

        logits, value, policy_prob = self._run_model(feature, legal_action)

        legal_action_np = np.array(legal_action, dtype=np.float32)
        epsilon = self._get_epsilon()

        # 构建探索分布: 合法动作均匀采样，并可偏置闪现动作。
        explore_prob = self._build_explore_distribution(legal_action_np)

        # 行为分布采用 policy 与 explore 的凸组合，避免 epsilon 分支导致 PPO old_prob 偏差。
        behavior_prob = (1.0 - epsilon) * policy_prob + epsilon * explore_prob
        behavior_prob = self._normalize_prob(behavior_prob)

        action = self._legal_sample(behavior_prob, use_max=False)
        d_action = self._legal_sample(policy_prob, use_max=True)
        self.predict_count += 1

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(behavior_prob),
                value=value,
            )
        ]

    def exploit(self, env_obs):
        """Greedy inference for evaluation.

        评估时贪心选择动作（利用）。
        """
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        """Train the model.

        训练模型。
        """
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        """Save model checkpoint.

        保存模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """Load model checkpoint.

        加载模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.logger.info(f"load model {model_file_path} successfully")

    def action_process(self, act_data, is_stochastic=True):
        """Unpack ActData to int action and update last_action.

        解包 ActData 为 int 动作并记录 last_action。
        """
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def _run_model(self, feature, legal_action):
        """Run model inference, return logits, value, prob.

        执行模型推理，返回 logits、value 和动作概率。
        """
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.array([feature]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, value = self.model(obs_tensor, inference=True)

        logits_np = logits.cpu().numpy()[0]
        value_np = value.cpu().numpy()[0]

        # Legal action masked softmax / 合法动作掩码 softmax
        legal_action_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits_np, legal_action_np)

        return logits_np, value_np, prob

    def _legal_soft_max(self, input_hidden, legal_action):
        """Softmax with legal action masking (numpy).

        合法动作掩码下的 softmax（numpy 版）。
        """
        _w, _e = 1e20, 1e-5
        tmp = input_hidden - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        """Sample action from probability distribution.

        按概率分布采样动作。
        """
        probs = self._normalize_prob(probs)

        if use_max:
            return int(np.argmax(probs))
        return int(np.random.choice(np.arange(Config.ACTION_NUM), p=probs))

    def _normalize_prob(self, probs):
        probs = np.array(probs, dtype=np.float32)
        probs = np.maximum(probs, 0.0)
        prob_sum = float(probs.sum())
        if prob_sum <= 1e-8:
            return np.ones((Config.ACTION_NUM,), dtype=np.float32) / float(Config.ACTION_NUM)
        return probs / prob_sum

    def _build_explore_distribution(self, legal_action):
        legal = np.array(legal_action, dtype=np.float32)
        legal = (legal > 0).astype(np.float32)
        if legal.sum() <= 0:
            legal[:] = 1.0

        explore = legal.copy()

        # 对合法闪现动作添加探索偏置。
        flash_legal = legal[8:]
        if flash_legal.sum() > 0 and np.random.rand() < self.phase_flash_explore_prob:
            explore[8:] += 1.0 * flash_legal

        return self._normalize_prob(explore)
