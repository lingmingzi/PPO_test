#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

PPO algorithm implementation for Gorge Chase PPO.
峡谷追猎 PPO 算法实现。

损失组成：
  total_loss = vf_coef * value_loss + policy_loss - beta * entropy_loss

  - value_loss  : Clipped value function loss（裁剪价值函数损失）
  - policy_loss : PPO Clipped surrogate objective（PPO 裁剪替代目标）
  - entropy_loss: Action entropy regularization（动作熵正则化，鼓励探索）
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from agent_ppo.conf.conf import Config


def _phase_to_stage(phase_name):
    phase_map = {
        "early": 1,
        "mid": 2,
        "late": 3,
    }
    return int(phase_map.get(str(phase_name).lower(), 0))


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM
        self.var_beta = Config.BETA_START
        self.beta_end = Config.BETA_END
        self.beta_decay = Config.BETA_DECAY
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM
        self.value_clip_param = Config.VALUE_CLIP_PARAM
        self.base_ppo_epochs = Config.PPO_EPOCHS
        self.base_target_kl = Config.TARGET_KL
        self.base_beta = Config.BETA_START
        self.ppo_epochs = self.base_ppo_epochs
        self.mini_batch_size = Config.PPO_MINI_BATCH_SIZE
        self.target_kl = self.base_target_kl
        self.phase_name = "early"
        self.beta_target = max(self.beta_end, self.base_beta)

        self.min_lr = Config.MIN_LEARNING_RATE
        self.lr_decay = Config.LR_DECAY
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)

        self.device_type = "cuda" if (self.device is not None and "cuda" in str(self.device).lower()) else "cpu"
        self.use_amp = bool(Config.ENABLE_AMP and self.device_type == "cuda")
        amp_dtype_name = str(Config.AMP_DTYPE).lower()
        self.amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
        self.use_grad_scaler = bool(self.use_amp and self.amp_dtype == torch.float16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_grad_scaler)

        self.last_report_monitor_time = 0
        self.train_step = 0

    def set_phase_params(self, phase_name=None, ppo_epochs=None, target_kl=None, beta_scale=None):
        if phase_name is not None:
            self.phase_name = str(phase_name)
        if ppo_epochs is not None:
            self.ppo_epochs = max(1, int(ppo_epochs))
        else:
            self.ppo_epochs = self.base_ppo_epochs
        if target_kl is not None:
            self.target_kl = max(float(target_kl), 1e-4)
        else:
            self.target_kl = self.base_target_kl

        if beta_scale is None:
            beta_scale = 1.0
        self.beta_target = max(self.beta_end, self.base_beta * float(max(beta_scale, 0.1)))
        if self.var_beta < self.beta_target:
            self.var_beta = self.beta_target

    def _stack_batch(self, values, dtype=torch.float32):
        first = values[0]
        if torch.is_tensor(first):
            batch = torch.stack(values, dim=0).to(self.device)
        else:
            batch = torch.as_tensor(np.stack(values, axis=0), device=self.device)
        return batch.to(dtype=dtype)

    def learn(self, list_sample_data):
        """Training entry: PPO update on a batch of SampleData.

        训练入口：对一批 SampleData 执行 PPO 更新。
        """
        if not list_sample_data:
            return

        obs = self._stack_batch([f.obs for f in list_sample_data], dtype=torch.float32)
        legal_action = self._stack_batch([f.legal_action for f in list_sample_data], dtype=torch.float32)
        act = self._stack_batch([f.act for f in list_sample_data], dtype=torch.float32).view(-1, 1).long()
        old_prob = self._stack_batch([f.prob for f in list_sample_data], dtype=torch.float32)
        reward = self._stack_batch([f.reward for f in list_sample_data], dtype=torch.float32)
        advantage = self._stack_batch([f.advantage for f in list_sample_data], dtype=torch.float32)
        old_value = self._stack_batch([f.value for f in list_sample_data], dtype=torch.float32)
        reward_sum = self._stack_batch([f.reward_sum for f in list_sample_data], dtype=torch.float32)

        invalid_rows = legal_action.sum(dim=1, keepdim=True) <= 0
        if invalid_rows.any():
            legal_action = legal_action.clone()
            legal_action[invalid_rows.expand_as(legal_action)] = 1.0

        old_prob = torch.clamp(old_prob, min=0.0)
        old_prob = old_prob / old_prob.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # 归一化优势，提升训练稳定性。
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)

        batch_size = obs.shape[0]
        mini_batch = max(1, min(self.mini_batch_size, batch_size))

        self.model.set_train_mode()

        loss_sum = 0.0
        value_loss_sum = 0.0
        policy_loss_sum = 0.0
        entropy_sum = 0.0
        approx_kl_sum = 0.0
        clip_frac_sum = 0.0
        update_cnt = 0
        early_stop = False

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_batch):
                mb_idx = indices[start : start + mini_batch]

                mb_obs = obs[mb_idx]
                mb_legal = legal_action[mb_idx]
                mb_act = act[mb_idx]
                mb_old_prob = old_prob[mb_idx]
                mb_adv = advantage[mb_idx]
                mb_old_value = old_value[mb_idx]
                mb_ret = reward_sum[mb_idx]
                mb_rew = reward[mb_idx]

                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device_type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    logits, value_pred = self.model(mb_obs)

                    total_loss, info_list = self._compute_loss(
                        logits=logits,
                        value_pred=value_pred,
                        legal_action=mb_legal,
                        old_action=mb_act,
                        old_prob=mb_old_prob,
                        advantage=mb_adv,
                        old_value=mb_old_value,
                        reward_sum=mb_ret,
                        reward=mb_rew,
                    )

                if self.use_grad_scaler:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
                    self.optimizer.step()

                loss_sum += float(total_loss.detach().item())
                value_loss_sum += float(info_list[0].detach().item())
                policy_loss_sum += float(info_list[1].detach().item())
                entropy_sum += float(info_list[2].detach().item())
                approx_kl_sum += float(info_list[3].detach().item())
                clip_frac_sum += float(info_list[4].detach().item())
                update_cnt += 1

                if float(info_list[3].detach().item()) > self.target_kl:
                    early_stop = True
                    break

            if early_stop:
                break

        if update_cnt <= 0:
            return

        if self.optimizer.param_groups[0]["lr"] > self.min_lr:
            self.scheduler.step()
            if self.optimizer.param_groups[0]["lr"] < self.min_lr:
                self.optimizer.param_groups[0]["lr"] = self.min_lr

        self.train_step += 1
        self.var_beta = max(self.beta_target, self.var_beta * self.beta_decay)

        with torch.no_grad():
            v_pred = old_value.view(-1)
            v_tgt = reward_sum.view(-1)
            var_y = torch.var(v_tgt)
            explained_var = torch.tensor(0.0, device=self.device)
            if torch.isfinite(var_y) and var_y > 1e-12:
                explained_var = 1.0 - torch.var(v_tgt - v_pred) / var_y

        mean_total_loss = loss_sum / update_cnt
        mean_value_loss = value_loss_sum / update_cnt
        mean_policy_loss = policy_loss_sum / update_cnt
        mean_entropy = entropy_sum / update_cnt
        mean_kl = approx_kl_sum / update_cnt
        mean_clip_frac = clip_frac_sum / update_cnt

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
            results = {
                "total_loss": round(mean_total_loss, 4),
                "value_loss": round(mean_value_loss, 4),
                "policy_loss": round(mean_policy_loss, 4),
                "entropy_loss": round(mean_entropy, 4),
                "approx_kl": round(mean_kl, 6),
                "clip_frac": round(mean_clip_frac, 4),
                "reward": round(reward.mean().item(), 4),
                "beta": round(float(self.var_beta), 6),
                "lr": round(lr, 8),
                "explained_var": round(float(explained_var.item()), 4),
                "ppo_updates": int(update_cnt),
                "phase": _phase_to_stage(self.phase_name),
            }
            self.logger.info(
                f"[train] total_loss:{results['total_loss']} "
                f"policy_loss:{results['policy_loss']} "
                f"value_loss:{results['value_loss']} "
                f"entropy:{results['entropy_loss']} "
                f"approx_kl:{results['approx_kl']} "
                f"clip_frac:{results['clip_frac']} "
                f"beta:{results['beta']} "
                f"exp_var:{results['explained_var']} "
                f"updates:{results['ppo_updates']}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

    def _compute_loss(
        self,
        logits,
        value_pred,
        legal_action,
        old_action,
        old_prob,
        advantage,
        old_value,
        reward_sum,
        reward,
    ):
        """Compute standard PPO loss (policy + value + entropy).

        计算标准 PPO 损失（策略损失 + 价值损失 + 熵正则化）。
        """
        # Masked softmax / 合法动作掩码 softmax
        prob_dist = self._masked_softmax(logits, legal_action)

        # Policy loss (PPO Clip) / 策略损失
        one_hot = F.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9)
        ratio = new_prob / old_action_prob
        adv = advantage.view(-1, 1)
        policy_loss1 = -ratio * adv
        policy_loss2 = -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()
        approx_kl = (torch.log(old_action_prob) - torch.log(new_prob.clamp(1e-9))).mean()
        clip_frac = (torch.abs(ratio - 1.0) > self.clip_param).float().mean()

        # Value loss (Clipped) / 价值损失
        vp = value_pred
        ov = old_value
        tdret = reward_sum
        value_clip = ov + (vp - ov).clamp(-self.value_clip_param, self.value_clip_param)
        value_loss = (
            0.5
            * torch.maximum(
                torch.square(tdret - vp),
                torch.square(tdret - value_clip),
            ).mean()
        )

        # Entropy loss / 熵损失
        entropy_loss = (-prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1).mean()

        # Total loss / 总损失
        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss

        return total_loss, [value_loss, policy_loss, entropy_loss, approx_kl, clip_frac]

    def _masked_softmax(self, logits, legal_action):
        """Softmax with legal action masking (suppress illegal actions).

        合法动作掩码下的 softmax（将非法动作概率压为极小值）。
        """
        legal = legal_action.float()
        invalid_rows = legal.sum(dim=1, keepdim=True) <= 0
        if invalid_rows.any():
            legal = legal.clone()
            legal[invalid_rows.expand_as(legal)] = 1.0

        masked_logits = logits.masked_fill(legal <= 0, -1e9)
        return F.softmax(masked_logits, dim=1)
