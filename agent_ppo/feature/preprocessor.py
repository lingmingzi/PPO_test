#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。
"""

import numpy as np
from agent_ppo.conf.conf import Config

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
MAX_SCORE_NORM = 600.0


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class Preprocessor:
    def __init__(self):
        self.phase_name = "early"
        self.flash_reward_scale = 1.0
        self.score_reward_scale = 1.0
        self.milestone_reward_scale = 1.0
        self._milestone_steps, self._milestone_rewards = self._build_step_milestone_schedule()
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 2000
        self.last_min_monster_dist_norm = 0.5
        self.prev_total_score = None
        self.prev_treasure_collected = 0
        self.prev_buff_collected = 0
        self.prev_hero_pos = None
        self.next_milestone_idx = 0
        self.last_milestone_reward = 0.0

    def set_phase_params(
        self,
        phase_name=None,
        flash_reward_scale=None,
        score_reward_scale=None,
        milestone_reward_scale=None,
    ):
        if phase_name is not None:
            self.phase_name = str(phase_name)
        if flash_reward_scale is not None:
            self.flash_reward_scale = float(max(flash_reward_scale, 0.1))
        if score_reward_scale is not None:
            self.score_reward_scale = float(max(score_reward_scale, 0.1))
        if milestone_reward_scale is not None:
            self.milestone_reward_scale = float(max(milestone_reward_scale, 0.1))

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _extract_observation(env_obs):
        if isinstance(env_obs, dict) and "observation" in env_obs:
            observation = env_obs.get("observation", {})
            return observation if isinstance(observation, dict) else {}, env_obs
        if isinstance(env_obs, dict):
            return env_obs, env_obs
        return {}, {}

    @staticmethod
    def _extract_hero(frame_state):
        heroes = frame_state.get("heroes", {}) if isinstance(frame_state, dict) else {}
        if isinstance(heroes, dict):
            return heroes
        if isinstance(heroes, list):
            return heroes[0] if heroes else {}
        return {}

    @staticmethod
    def _extract_monsters(frame_state):
        monsters = frame_state.get("monsters", []) if isinstance(frame_state, dict) else []
        if isinstance(monsters, dict):
            return list(monsters.values())
        if isinstance(monsters, list):
            return monsters
        return []

    @staticmethod
    def _to_map_array(map_info):
        if map_info is None:
            return None
        try:
            if isinstance(map_info, list) and map_info and isinstance(map_info[0], dict):
                return np.array([line.get("values", []) for line in map_info], dtype=np.float32)
            return np.asarray(map_info, dtype=np.float32)
        except (TypeError, ValueError):
            return None

    def _build_step_milestone_schedule(self):
        if not bool(getattr(Config, "ENABLE_STEP_MILESTONE_REWARD", False)):
            return [], []

        raw_steps = getattr(Config, "STEP_MILESTONE_STEPS", [])
        raw_rewards = getattr(Config, "STEP_MILESTONE_REWARDS", [])

        try:
            steps = [int(x) for x in raw_steps]
        except (TypeError, ValueError):
            return [], []

        steps = [s for s in steps if s > 0]
        if not steps:
            return [], []

        try:
            rewards = [float(x) for x in raw_rewards]
        except (TypeError, ValueError):
            return [], []

        if not rewards:
            return [], []

        if len(rewards) < len(steps):
            rewards = rewards + [rewards[-1]] * (len(steps) - len(rewards))
        elif len(rewards) > len(steps):
            rewards = rewards[: len(steps)]

        # 去重并按步数排序，同步保留对应奖励。
        milestone_map = {}
        for step, reward in zip(steps, rewards):
            if reward > 0:
                milestone_map[int(step)] = float(reward)

        if not milestone_map:
            return [], []

        ordered = sorted(milestone_map.items(), key=lambda x: x[0])
        return [x[0] for x in ordered], [x[1] for x in ordered]

    def _parse_legal_action(self, legal_act_raw):
        mask = np.ones((Config.ACTION_NUM,), dtype=np.float32)

        if legal_act_raw is None:
            return mask.tolist()

        # dict 格式: {move: [...], talent/flash: [...]} 或分支开关
        if isinstance(legal_act_raw, dict):
            move_raw = None
            flash_raw = None
            for key in ("move", "move_action", "move_legal", "move_mask", "direction"):
                if key in legal_act_raw:
                    move_raw = legal_act_raw.get(key)
                    break
            for key in ("talent", "flash", "talent_action", "talent_legal", "talent_mask"):
                if key in legal_act_raw:
                    flash_raw = legal_act_raw.get(key)
                    break

            if move_raw is not None:
                arr = np.asarray(move_raw).reshape(-1)
                if arr.size >= 8:
                    mask[:8] = arr[:8].astype(np.float32)
                elif arr.size == 1 and not bool(arr[0]):
                    mask[:8] = 0.0

            if flash_raw is not None:
                arr = np.asarray(flash_raw).reshape(-1)
                if arr.size >= 8:
                    mask[8:] = arr[:8].astype(np.float32)
                elif arr.size == 1:
                    if bool(arr[0]):
                        mask[8:] = mask[:8]
                    else:
                        mask[8:] = 0.0

        elif isinstance(legal_act_raw, (list, tuple, np.ndarray)):
            arr = np.asarray(legal_act_raw).reshape(-1)

            if arr.size >= Config.ACTION_NUM:
                mask = arr[: Config.ACTION_NUM].astype(np.float32)
            elif arr.size == 9:
                mask[:8] = arr[:8].astype(np.float32)
                mask[8:] = mask[:8] if bool(arr[8]) else 0.0
            elif arr.size == 8:
                mask[:8] = arr.astype(np.float32)
                mask[8:] = arr.astype(np.float32)
            elif arr.size >= 2:
                # 常见分支开关: [move_enable, flash_enable]
                if not bool(arr[0]):
                    mask[:8] = 0.0
                if bool(arr[1]):
                    mask[8:] = mask[:8]
                else:
                    mask[8:] = 0.0
            else:
                # 可能是动作id列表
                try:
                    actions = [int(x) for x in legal_act_raw]
                    if actions:
                        mask[:] = 0.0
                        for a in actions:
                            if 0 <= a < Config.ACTION_NUM:
                                mask[a] = 1.0
                except (TypeError, ValueError):
                    pass

        # 兜底，避免全非法导致采样失败
        if float(mask.sum()) <= 0:
            mask[:] = 1.0

        return mask.tolist()

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, and reward.

        将 env_obs 转换为特征向量、合法动作掩码和即时奖励。
        """
        observation, raw_env = self._extract_observation(env_obs)
        frame_state = observation.get("frame_state", {})
        env_info = observation.get("env_info", {})
        if not isinstance(env_info, dict):
            env_info = {}

        if isinstance(raw_env, dict):
            raw_env_info = raw_env.get("env_info")
            if isinstance(raw_env_info, dict):
                merged = dict(raw_env_info)
                merged.update(env_info)
                env_info = merged

        map_info = observation.get("map_info", raw_env.get("map_info", [])) if isinstance(raw_env, dict) else []
        legal_act_raw = observation.get("legal_act", observation.get("legal_action"))
        if legal_act_raw is None and isinstance(raw_env, dict):
            legal_act_raw = raw_env.get("legal_act", raw_env.get("legal_action"))

        self.step_no = self._safe_int(observation.get("step_no", env_info.get("step_no", self.step_no + 1)), self.step_no + 1)
        self.max_step = max(1, self._safe_int(env_info.get("max_step", self.max_step), self.max_step))

        hero = self._extract_hero(frame_state)
        hero_pos = hero.get("pos", {}) if isinstance(hero, dict) else {}
        hero_x = self._safe_float(hero_pos.get("x", hero_pos.get("X", 0.0)), 0.0)
        hero_z = self._safe_float(hero_pos.get("z", hero_pos.get("y", 0.0)), 0.0)
        hero_xy = np.array([hero_x, hero_z], dtype=np.float32)

        legal_action = self._parse_legal_action(legal_act_raw)
        legal_action_np = np.array(legal_action, dtype=np.float32)
        flash_available = float(legal_action_np[8:].sum() > 0)

        flash_cd_cfg = self._safe_float(env_info.get("flash_cooldown", MAX_FLASH_CD), MAX_FLASH_CD)
        buff_remain = self._safe_float(
            hero.get("buff_remain_time", hero.get("buff_remaining_time", 0.0)) if isinstance(hero, dict) else 0.0,
            0.0,
        )
        has_buff = float(buff_remain > 0.0)
        step_norm = _norm(self.step_no, float(self.max_step))

        # Hero self features (6D)
        hero_feat = np.array(
            [
                _norm(hero_x, MAP_SIZE),
                _norm(hero_z, MAP_SIZE),
                flash_available,
                _norm(flash_cd_cfg, MAX_FLASH_CD),
                has_buff,
                step_norm,
            ],
            dtype=np.float32,
        )

        # Monster features (5D x 2)
        monsters = self._extract_monsters(frame_state)
        monster_feats = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                if not isinstance(m, dict):
                    monster_feats.append(np.zeros(5, dtype=np.float32))
                    continue

                m_pos = m.get("pos", {})
                has_pos = isinstance(m_pos, dict) and ("x" in m_pos) and ("z" in m_pos or "y" in m_pos)
                is_in_view = float(m.get("is_in_view", 1 if has_pos else 0))

                dist_bucket = self._safe_float(m.get("hero_l2_distance", -1), -1)
                if dist_bucket >= 0:
                    dist_norm = float(np.clip(dist_bucket / 5.0, 0.0, 1.0))
                elif has_pos:
                    mx = self._safe_float(m_pos.get("x", 0.0), 0.0)
                    mz = self._safe_float(m_pos.get("z", m_pos.get("y", 0.0)), 0.0)
                    raw_dist = float(np.linalg.norm(np.array([hero_x - mx, hero_z - mz], dtype=np.float32)))
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                else:
                    dist_norm = 1.0

                dir_norm = _norm(self._safe_float(m.get("hero_relative_direction", 0), 0.0), 8.0)
                speed_norm = _norm(self._safe_float(m.get("speed", 1), 1.0), MAX_MONSTER_SPEED)
                threat = float(1.0 - dist_norm) if is_in_view > 0 else 0.0

                monster_feats.append(
                    np.array([is_in_view, dist_norm, dir_norm, speed_norm, threat], dtype=np.float32)
                )
            else:
                monster_feats.append(np.zeros(5, dtype=np.float32))

        # Local map features (16D)
        map_feat = np.zeros(16, dtype=np.float32)
        map_arr = self._to_map_array(map_info)
        if map_arr is not None and map_arr.ndim == 2 and map_arr.size > 0:
            map_arr = np.clip(map_arr, 0, 1)
            h, w = map_arr.shape
            center_h = h // 2
            center_w = w // 2

            top = center_h - 2
            left = center_w - 2
            bottom = top + 4
            right = left + 4

            src_x0 = max(0, top)
            src_x1 = min(h, bottom)
            src_y0 = max(0, left)
            src_y1 = min(w, right)

            dst_x0 = src_x0 - top
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            dst_y0 = src_y0 - left
            dst_y1 = dst_y0 + (src_y1 - src_y0)

            patch = np.zeros((4, 4), dtype=np.float32)
            patch[dst_x0:dst_x1, dst_y0:dst_y1] = map_arr[src_x0:src_x1, src_y0:src_y1]
            map_feat = patch.reshape(-1)

        total_score = self._safe_float(env_info.get("total_score", 0.0), 0.0)
        treasure_collected = self._safe_int(
            hero.get("treasure_collected_count", env_info.get("treasures_collected", 0)) if isinstance(hero, dict) else 0,
            0,
        )
        buff_collected = self._safe_int(env_info.get("collected_buff", 0), 0)

        # Progress features (4D)
        progress_feat = np.array(
            [
                step_norm,
                _norm(total_score, MAX_SCORE_NORM),
                _norm(treasure_collected, 10.0),
                _norm(buff_collected, 2.0),
            ],
            dtype=np.float32,
        )

        # Concatenate features / 拼接特征
        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                map_feat,
                legal_action_np,
                progress_feat,
            ]
        )
        assert feature.shape[0] == Config.DIM_OF_OBSERVATION, (
            f"feature dim mismatch: got {feature.shape[0]}, expect {Config.DIM_OF_OBSERVATION}"
        )

        # Step reward / 即时奖励
        cur_min_dist_norm = 1.0
        for m_feat in monster_feats:
            if m_feat[0] > 0:
                cur_min_dist_norm = min(cur_min_dist_norm, m_feat[1])

        dist_shaping = Config.REW_MONSTER_DIST * (cur_min_dist_norm - self.last_min_monster_dist_norm)
        survive_reward = Config.REW_STEP_SURVIVE

        self.last_min_monster_dist_norm = cur_min_dist_norm

        if self.prev_total_score is None:
            score_delta = 0.0
        else:
            score_delta = total_score - self.prev_total_score

        treasure_get = max(0, treasure_collected - self.prev_treasure_collected)
        buff_get = max(0, buff_collected - self.prev_buff_collected)

        reward_value = survive_reward
        reward_value += Config.REW_SCORE_SCALE * self.score_reward_scale * score_delta
        reward_value += dist_shaping
        reward_value += treasure_get * Config.REW_TREASURE_GET
        reward_value += buff_get * Config.REW_BUFF_GET

        if isinstance(last_action, int) and last_action >= 8:
            reward_value += Config.REW_FLASH_CAST * self.flash_reward_scale
            if self.prev_hero_pos is not None:
                displacement = float(np.linalg.norm(hero_xy - self.prev_hero_pos))
                if displacement >= 6.0:
                    reward_value += Config.REW_FLASH_EFFECTIVE * self.flash_reward_scale

        # Step milestone sparse reward / 步数里程碑稀疏奖励（每局每里程碑仅触发一次）
        milestone_bonus = 0.0
        while self.next_milestone_idx < len(self._milestone_steps):
            if self.step_no < self._milestone_steps[self.next_milestone_idx]:
                break
            milestone_bonus += self._milestone_rewards[self.next_milestone_idx]
            self.next_milestone_idx += 1

        milestone_bonus *= self.milestone_reward_scale
        reward_value += milestone_bonus
        self.last_milestone_reward = milestone_bonus

        terminated = bool(env_obs.get("terminated", False)) if isinstance(env_obs, dict) else False
        truncated = bool(env_obs.get("truncated", False)) if isinstance(env_obs, dict) else False
        if terminated:
            reward_value += Config.REW_TERMINATED
        if truncated:
            reward_value += Config.REW_TRUNCATED

        self.prev_total_score = total_score
        self.prev_treasure_collected = treasure_collected
        self.prev_buff_collected = buff_collected
        self.prev_hero_pos = hero_xy

        reward = [float(np.float32(reward_value))]

        return feature, legal_action, reward
