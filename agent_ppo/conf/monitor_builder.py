#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Monitor panel configuration builder for Gorge Chase.
峡谷追猎监控面板配置构建器。
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("峡谷追猎")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="累积回报",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="avg(reward{})",
        )
        .end_panel()
        .add_panel(
            name="里程碑奖励",
            name_en="milestone_bonus",
            type="line",
        )
        .add_metric(
            metrics_name="milestone_bonus",
            expr="avg(milestone_bonus{})",
        )
        .end_panel()
        .add_panel(
            name="里程碑命中数",
            name_en="milestone_hits",
            type="line",
        )
        .add_metric(
            metrics_name="milestone_hits",
            expr="avg(milestone_hits{})",
        )
        .end_panel()
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="avg(total_loss{})",
        )
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="avg(value_loss{})",
        )
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="avg(policy_loss{})",
        )
        .end_panel()
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="avg(entropy_loss{})",
        )
        .end_panel()
        .end_group()
        .add_group(
            group_name="任务指标",
            group_name_en="character_metrics",
        )
        .add_panel(
            name="任务总分",
            name_en="task_total_score",
            type="line",
        )
        .add_metric(
            metrics_name="task_total_score",
            expr="avg(task_total_score{})",
        )
        .end_panel()
        .add_panel(
            name="步数分",
            name_en="step_score",
            type="line",
        )
        .add_metric(
            metrics_name="step_score",
            expr="avg(step_score{})",
        )
        .end_panel()
        .add_panel(
            name="宝箱分",
            name_en="treasure_score",
            type="line",
        )
        .add_metric(
            metrics_name="treasure_score",
            expr="avg(treasure_score{})",
        )
        .end_panel()
        .add_panel(
            name="总步数",
            name_en="total_steps",
            type="line",
        )
        .add_metric(
            metrics_name="total_steps",
            expr="avg(total_steps{})",
        )
        .end_panel()
        .add_panel(
            name="宝箱数",
            name_en="treasure_count",
            type="line",
        )
        .add_metric(
            metrics_name="treasure_count",
            expr="avg(treasure_count{})",
        )
        .end_panel()
        .add_panel(
            name="闪现数",
            name_en="flash_count",
            type="line",
        )
        .add_metric(
            metrics_name="flash_count",
            expr="avg(flash_count{})",
        )
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
