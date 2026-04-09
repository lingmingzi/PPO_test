# Gorge Chase PPO Overview

## 1. Game Basics

This PPO agent is built for the Gorge Chase task.

Core environment facts (from train_env_conf.toml):

- Map IDs used for training: 1 to 10
- Map sampling mode: sequential (map_random = false)
- Treasure count: 10
- Buff count: 2
- Buff respawn cooldown: 200 steps
- Talent cooldown: 100 steps
- Second monster spawn interval: 300 steps
- Monster speedup step: 500
- Max steps per episode: 1000

Task score decomposition used in current monitor logic:

- task_total_score ~= step_score + treasure_score
- step_score = finished_steps * 1.5
- treasure_score = treasure_count * 100

Action and observation settings:

- Action space: 16 actions (8 move + 8 flash)
- Observation feature length: 52
- Value head count: 1

## 2. PPO Package Structure

- agent_ppo/agent.py
  - Main agent class.
  - Handles inference, exploration, save/load model, and delegates training to algorithm.
- agent_ppo/model/model.py
  - Network definition.
  - Feature-group encoders + shared trunk + actor/critic heads.
- agent_ppo/algorithm/algorithm.py
  - PPO learning implementation.
  - Clipped policy loss, clipped value loss, entropy regularization, KL-based early stop.
- agent_ppo/feature/preprocessor.py
  - Feature extraction and reward shaping.
  - Supports robust field parsing for env payload variants.
- agent_ppo/feature/definition.py
  - Data classes and sample processing.
  - GAE computation and reward_sum construction.
- agent_ppo/workflow/train_workflow.py
  - Episode loop, sample collection, phase scheduling, monitor reporting.
- agent_ppo/conf/conf.py
  - Hyperparameters, stage schedule, reward coefficients.
- agent_ppo/conf/monitor_builder.py
  - Monitor panel definitions for algorithm metrics and task metrics.

## 3. End-to-End Training Flow

1. Entry
- train_test.py calls run_train_test with algorithm_name = ppo.
- Runtime env vars include replay buffer settings and dump_model_freq.
- Optional resume logic restores latest checkpoint into agent_ppo/ckpt.

2. Framework startup
- Learner and AiSrv processes are started by framework utilities.
- Policy configured as async builder with algo = ppo.

3. Workflow loop (train_workflow.py)
- Read train env config.
- Initialize EpisodeRunner.
- Repeatedly run episodes and collect SampleData frames.

4. Per-step interaction
- observation_process: env obs -> feature + legal action + reward.
- predict: actor forward pass + epsilon mixed exploration distribution.
- action_process: selected action -> env.step(action).
- Build frame samples (obs, legal_action, act, reward, value, prob, done).

5. Sample post-processing
- sample_process fills next_value.
- GAE computes advantage and reward_sum.

6. Learning
- In distributed wrapper mode, samples are sent to learner.
- Learner runs PPO updates with:
  - mini-batch optimization
  - mixed precision (when enabled)
  - gradient clipping
  - learning-rate decay
  - KL threshold early stopping

7. Checkpoint and monitor
- Workflow calls periodic save_model every 1800 seconds.
- Framework also handles model dump by dump_model_freq.
- Monitor data is pushed every 60 seconds on episode end.

## 4. Three-Stage Training Schedule

Stage boundaries (by train_global_step):

- Early: step < 30000
- Mid: 30000 <= step < 120000
- Late: step >= 120000

Per-stage parameters:

| Parameter | Early | Mid | Late |
|---|---:|---:|---:|
| ratio_low | 1.0 | 1.5 | 2.0 |
| ratio_high | 1.5 | 2.0 | 2.5 |
| eps_scale | 1.15 | 1.00 | 0.85 |
| flash_explore_prob | 0.55 | 0.35 | 0.20 |
| ppo_epochs | 3 | 4 | 5 |
| target_kl | 0.04 | 0.03 | 0.02 |
| beta_scale | 1.20 | 1.00 | 0.80 |
| flash_reward_scale | 1.60 | 1.20 | 1.00 |
| score_reward_scale | 0.90 | 1.00 | 1.10 |
| milestone_reward_scale | 1.10 | 1.00 | 0.90 |

How stage is applied:

- Agent: phase_name, epsilon_scale, flash exploration probability.
- Algorithm: ppo_epochs, target_kl, beta target scale.
- Preprocessor: flash/score/milestone reward scales.

## 5. Reward Function Design

Reward is computed in preprocessor.feature_process.

Base components:

- Survival reward: REW_STEP_SURVIVE
- Score delta reward: REW_SCORE_SCALE * score_reward_scale * score_delta
- Monster distance shaping: REW_MONSTER_DIST * (cur_min_dist_norm - prev_min_dist_norm)
- Treasure gain reward: treasure_get * REW_TREASURE_GET
- Buff gain reward: buff_get * REW_BUFF_GET

Flash-related components:

- Flash cast reward when last_action >= 8: REW_FLASH_CAST * flash_reward_scale
- Effective flash reward if displacement >= 6.0: REW_FLASH_EFFECTIVE * flash_reward_scale

Milestone sparse reward:

- Milestone steps: [300, 600, 900, 1200, 1500, 1800]
- Milestone rewards: [0.06, 0.08, 0.10, 0.12, 0.14, 0.16]
- Applied once per milestone per episode, then scaled by milestone_reward_scale.

Terminal adjustments:

- terminated: + REW_TERMINATED (-1.0)
- truncated: + REW_TRUNCATED (+1.0)

Compact form:

reward_t = survive + score_delta_term + distance_term + treasure_term + buff_term + flash_terms + milestone_term + terminal_term

## 6. Monitor Metrics (Current)

Algorithm metrics include:

- reward
- total_loss, value_loss, policy_loss, entropy_loss
- approx_kl, clip_frac
- beta, lr, explained_var
- ppo_updates
- phase (numeric stage id: 1/2/3)

Task metrics include:

- task_total_score
- step_score
- treasure_score
- total_steps
- treasure_count
- flash_count
- episode_steps, episode_cnt
- milestone_hits, milestone_bonus

## 7. Model Save and Load Behavior

Current behavior combines both framework-level and workflow-level saving:

- Workflow periodic save: every 1800 seconds, call agent.save_model().
- Framework save: controlled by dump_model_freq from run_train_test env vars.
- Periodic model refresh in runner: agent.load_model(id="latest") every LOAD_MODEL_INTERVAL_SEC (min 5 sec).
- Optional resume at startup: latest backup checkpoint can be restored to agent_ppo/ckpt and loaded via preload settings.

## 8. Quick Practical Notes

- If training is unstable, first tune target_kl, ppo_epochs, and beta_scale by stage.
- If exploration is weak, increase flash_explore_prob and eps_scale in early stage.
- If reward drifts from task objective, focus on score/treasure-related terms and milestone scales.
- If monitor lacks expected values, check env field names and fallback extraction logic in workflow/preprocessor.
