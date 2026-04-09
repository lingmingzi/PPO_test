#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase PPO.
峡谷追猎 PPO 训练工作流。
"""

import os
import time

import numpy as np
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_observation_and_env_info(env_obs):
    if not isinstance(env_obs, dict):
        return {}, {}

    observation = env_obs.get("observation", env_obs)
    if not isinstance(observation, dict):
        observation = {}

    env_info = observation.get("env_info", {})
    if not isinstance(env_info, dict):
        env_info = {}

    raw_env_info = env_obs.get("env_info")
    if isinstance(raw_env_info, dict):
        merged = dict(raw_env_info)
        merged.update(env_info)
        env_info = merged

    return observation, env_info


def _get_treasure_collected_count(observation, env_info):
    search_dicts = []
    if isinstance(observation, dict):
        score_info = observation.get("score_info", {})
        if isinstance(score_info, dict):
            search_dicts.append(score_info)

    if isinstance(env_info, dict):
        search_dicts.append(env_info)
        for key in ("treasure", "treasure_info"):
            treasure_info = env_info.get(key)
            if isinstance(treasure_info, dict):
                search_dicts.append(treasure_info)

    for score_dict in search_dicts:
        for key in (
            "treasure_collected_count",
            "treasure_count",
            "collected_treasures",
            "treasure_num",
            "treasure_collected",
            "treasures_collected",
        ):
            if key in score_dict:
                return _safe_int(score_dict.get(key), 0)
    return 0


def _get_official_total_score(observation, env_info):
    if isinstance(env_info, dict) and "total_score" in env_info:
        return _safe_float(env_info.get("total_score"), 0.0), True

    if isinstance(observation, dict):
        score_info = observation.get("score_info", {})
        if isinstance(score_info, dict) and "total_score" in score_info:
            return _safe_float(score_info.get("total_score"), 0.0), True

    return 0.0, False


def _calc_episode_objective_score(step_no, observation, env_info, official_total_score=None):
    # Match official eval decomposition: total = step_score + treasure_score
    finished_steps = _safe_int(step_no, 0)
    treasure_collected = _get_treasure_collected_count(observation, env_info)

    step_score = float(finished_steps) * 1.5
    if official_total_score is not None:
        parsed_total = step_score + float(treasure_collected) * 100.0
        if abs(_safe_float(official_total_score, 0.0) - parsed_total) > 25.0:
            score_delta = max(0.0, _safe_float(official_total_score, 0.0) - step_score)
            treasure_collected = max(0, int(round(score_delta / 100.0)))

    treasure_score = float(treasure_collected) * 100.0
    objective_score = step_score + treasure_score
    return objective_score, step_score, treasure_score, finished_steps, treasure_collected


def _resolve_training_phase(train_global_step):
    step = max(_safe_int(train_global_step, 0), 0)
    if step < Config.TRAIN_PHASE_STAGE1_END_STEP:
        return {
            "phase": "early",
            "ratio_low": Config.TRAIN_PHASE_STAGE1_RATIO_LOW,
            "ratio_high": Config.TRAIN_PHASE_STAGE1_RATIO_HIGH,
            "eps_scale": Config.TRAIN_PHASE_STAGE1_EPS_SCALE,
            "flash_explore_prob": Config.TRAIN_PHASE_STAGE1_FLASH_EXPLORE_PROB,
            "ppo_epochs": Config.TRAIN_PHASE_STAGE1_PPO_EPOCHS,
            "target_kl": Config.TRAIN_PHASE_STAGE1_TARGET_KL,
            "beta_scale": Config.TRAIN_PHASE_STAGE1_BETA_SCALE,
            "flash_reward_scale": Config.TRAIN_PHASE_STAGE1_FLASH_REWARD_SCALE,
            "score_reward_scale": Config.TRAIN_PHASE_STAGE1_SCORE_REWARD_SCALE,
            "milestone_reward_scale": Config.TRAIN_PHASE_STAGE1_MILESTONE_REWARD_SCALE,
        }
    if step < Config.TRAIN_PHASE_STAGE2_END_STEP:
        return {
            "phase": "mid",
            "ratio_low": Config.TRAIN_PHASE_STAGE2_RATIO_LOW,
            "ratio_high": Config.TRAIN_PHASE_STAGE2_RATIO_HIGH,
            "eps_scale": Config.TRAIN_PHASE_STAGE2_EPS_SCALE,
            "flash_explore_prob": Config.TRAIN_PHASE_STAGE2_FLASH_EXPLORE_PROB,
            "ppo_epochs": Config.TRAIN_PHASE_STAGE2_PPO_EPOCHS,
            "target_kl": Config.TRAIN_PHASE_STAGE2_TARGET_KL,
            "beta_scale": Config.TRAIN_PHASE_STAGE2_BETA_SCALE,
            "flash_reward_scale": Config.TRAIN_PHASE_STAGE2_FLASH_REWARD_SCALE,
            "score_reward_scale": Config.TRAIN_PHASE_STAGE2_SCORE_REWARD_SCALE,
            "milestone_reward_scale": Config.TRAIN_PHASE_STAGE2_MILESTONE_REWARD_SCALE,
        }
    return {
        "phase": "late",
        "ratio_low": Config.TRAIN_PHASE_STAGE3_RATIO_LOW,
        "ratio_high": Config.TRAIN_PHASE_STAGE3_RATIO_HIGH,
        "eps_scale": Config.TRAIN_PHASE_STAGE3_EPS_SCALE,
        "flash_explore_prob": Config.TRAIN_PHASE_STAGE3_FLASH_EXPLORE_PROB,
        "ppo_epochs": Config.TRAIN_PHASE_STAGE3_PPO_EPOCHS,
        "target_kl": Config.TRAIN_PHASE_STAGE3_TARGET_KL,
        "beta_scale": Config.TRAIN_PHASE_STAGE3_BETA_SCALE,
        "flash_reward_scale": Config.TRAIN_PHASE_STAGE3_FLASH_REWARD_SCALE,
        "score_reward_scale": Config.TRAIN_PHASE_STAGE3_SCORE_REWARD_SCALE,
        "milestone_reward_scale": Config.TRAIN_PHASE_STAGE3_MILESTONE_REWARD_SCALE,
    }


def _apply_phase(agent, phase_cfg):
    phase_name = phase_cfg["phase"]

    if hasattr(agent, "set_phase_params"):
        agent.set_phase_params(
            phase_name=phase_name,
            epsilon_scale=phase_cfg["eps_scale"],
            flash_explore_prob=phase_cfg["flash_explore_prob"],
        )
    else:
        setattr(agent, "phase_name", phase_name)
        setattr(agent, "epsilon_scale", float(phase_cfg["eps_scale"]))
        setattr(agent, "phase_flash_explore_prob", float(phase_cfg["flash_explore_prob"]))

    if hasattr(agent, "algorithm") and hasattr(agent.algorithm, "set_phase_params"):
        agent.algorithm.set_phase_params(
            phase_name=phase_name,
            ppo_epochs=phase_cfg["ppo_epochs"],
            target_kl=phase_cfg["target_kl"],
            beta_scale=phase_cfg["beta_scale"],
        )

    if hasattr(agent, "preprocessor") and hasattr(agent.preprocessor, "set_phase_params"):
        agent.preprocessor.set_phase_params(
            phase_name=phase_name,
            flash_reward_scale=phase_cfg["flash_reward_scale"],
            score_reward_scale=phase_cfg["score_reward_scale"],
            milestone_reward_scale=phase_cfg["milestone_reward_scale"],
        )


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    # Read user config / 读取用户配置
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            if not g_data:
                continue
            if hasattr(agent, "send_sample_data"):
                agent.send_sample_data(g_data)
            else:
                agent.learn(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0
        self.last_load_model_time = 0.0
        self.load_model_interval_sec = max(float(os.environ.get("LOAD_MODEL_INTERVAL_SEC", "5")), 5.0)
        self.phase_name = None

        init_phase = _resolve_training_phase(0)
        _apply_phase(self.agent, init_phase)
        self.phase_name = init_phase["phase"]

    def run_episodes(self):
        """Run a single episode and yield collected samples.

        执行单局对局并 yield 训练样本。
        """
        while True:
            # Periodically fetch training metrics / 定期获取训练指标
            now = time.time()
            if now - self.last_get_training_metrics_time >= 30:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    basic = training_metrics.get("basic", {}) if isinstance(training_metrics, dict) else {}
                    train_global_step = _safe_float(basic.get("train_global_step", 0.0), 0.0)
                    ratio = _safe_float(basic.get("sample_production_and_consumption_ratio", 0.0), 0.0)
                    phase_cfg = _resolve_training_phase(train_global_step)
                    _apply_phase(self.agent, phase_cfg)

                    if self.phase_name != phase_cfg["phase"]:
                        self.phase_name = phase_cfg["phase"]
                        self.logger.info(
                            "training phase switched => "
                            f"phase:{self.phase_name}, global_step:{train_global_step:.0f}, "
                            f"target_ratio:[{phase_cfg['ratio_low']:.2f}, {phase_cfg['ratio_high']:.2f}], "
                            f"eps_scale:{phase_cfg['eps_scale']:.2f}, "
                            f"flash_explore_prob:{phase_cfg['flash_explore_prob']:.2f}, "
                            f"ppo_epochs:{phase_cfg['ppo_epochs']}, target_kl:{phase_cfg['target_kl']:.3f}"
                        )

                    self.logger.info(
                        "training health => "
                        f"global_step:{train_global_step:.0f}, prod_cons_ratio:{ratio:.3f}, "
                        f"phase:{phase_cfg['phase']}, "
                        f"target_ratio:[{phase_cfg['ratio_low']:.2f}, {phase_cfg['ratio_high']:.2f}]"
                    )

            # Reset env / 重置环境
            env_obs = self.env.reset(self.usr_conf)

            # Disaster recovery / 容灾处理
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # Reset agent & load latest model / 重置 Agent 并加载最新模型
            self.agent.reset(env_obs)
            now = time.time()
            if now - self.last_load_model_time >= self.load_model_interval_sec:
                self.agent.load_model(id="latest")
                self.last_load_model_time = now

            # Initial observation / 初始观测处理
            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0
            flash_count = 0

            while not done:
                # Predict action / Agent 推理（随机采样）
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)
                if _safe_int(act, -1) >= 8:
                    flash_count += 1

                # Step env / 与环境交互
                env_reward, env_obs = self.env.step(act)

                # Disaster recovery / 容灾处理
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                # Next observation / 处理下一步观测
                _obs_data, _remain_info = self.agent.observation_process(env_obs)

                # Step reward / 每步即时奖励
                reward = np.array(_remain_info.get("reward", [0.0]), dtype=np.float32)
                total_reward += float(reward[0])

                milestone_bonus = 0.0
                milestone_hits = 0
                if hasattr(self.agent, "preprocessor"):
                    milestone_bonus = float(getattr(self.agent.preprocessor, "last_milestone_reward", 0.0))
                    milestone_hits = int(getattr(self.agent.preprocessor, "next_milestone_idx", 0))
                if milestone_bonus > 0:
                    self.logger.info(
                        f"[milestone] episode:{self.episode_cnt} step:{step} "
                        f"bonus:{milestone_bonus:.4f} hits:{milestone_hits}"
                    )

                # Terminal reward / 终局奖励
                if done:
                    observation, env_info = _extract_observation_and_env_info(env_obs)
                    official_total_score, has_official_total = _get_official_total_score(observation, env_info)
                    objective_score, step_score, treasure_score, finished_steps, treasure_collected = (
                        _calc_episode_objective_score(
                            step_no=step,
                            observation=observation,
                            env_info=env_info,
                            official_total_score=official_total_score,
                        )
                    )
                    total_score = official_total_score if has_official_total else objective_score

                    if terminated:
                        result_str = "FAIL"
                    else:
                        result_str = "WIN"

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"step_score:{step_score:.1f} treasure_score:{treasure_score:.1f} "
                        f"treasure_count:{treasure_collected} flash_count:{flash_count} "
                        f"total_reward:{total_reward:.3f}"
                    )

                # Build sample frame / 构造样本帧
                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                # Episode end / 对局结束
                if done:
                    # Monitor report / 监控上报
                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        monitor_data = {
                            "reward": round(total_reward, 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                            "milestone_hits": milestone_hits,
                            "milestone_bonus": round(milestone_bonus, 4),
                            "task_total_score": round(total_score, 4),
                            "step_score": round(step_score, 4),
                            "treasure_score": round(treasure_score, 4),
                            "total_steps": int(finished_steps),
                            "treasure_count": int(treasure_collected),
                            "flash_count": int(flash_count),
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                # Update state / 状态更新
                obs_data = _obs_data
                remain_info = _remain_info
