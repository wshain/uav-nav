# 论文主干思路（Master Thesis Backbone）

> 维护说明：本文件记录学位论文的**研究主线、章节骨架、实验设计与论证逻辑**。每次新增实验、图表或引用，须同步更新本文件、`CITATION_REGISTRY.md` 与 `references.bib`。

---

## 基本信息

| 字段              | 内容                                       |
| ----------------- | ------------------------------------------ |
| 论文题目          | 基于深度强化学习的多目标无人机路径导航研究 |
| 学位论文/实践成果 | 实践成果                                   |
| 学生              | 王世豪（24S136052）                        |
| 导师              | 李治军                                     |
| 学院              | 计算学部                                   |
| 学科              | 电子信息                                   |
| 开题日期          | 2025-09-12                                 |
| 中期检查          | 2026-06                                    |

---

## 一句话主线

面向农业巡检、工业运维等多目标作业场景，构建**上层 TSP 序列规划 + 下层 AirSim 强化学习导航避障**的两层系统：上层用指针网络快速近似最优访问顺序，下层用 SAC 等 DRL 策略实现连续空间避障飞行；联合评测揭示连续任务显著拉低成功率、上下层规划—执行割裂，且无人机作业中**整链成功率较路径几何最优更具优先级**；**中期之后**在 TSP 序列训练中引入障碍物惩罚，使上层规划与下层可飞性对齐。

---

## 研究路径（阶段划分）

| 阶段                     | 内容                                                         | 状态 |
| ------------------------ | ------------------------------------------------------------ | ---- |
| **中期（已完成）**       | 上层无障碍 TSP 指针网络训练与 500 实例 ratio 评测            | ✅   |
|                          | 下层 AirSim SimpleAvoid 中 SAC/TD3/PPO 导航避障 benchmark    | ✅   |
|                          | 上下层初步联合评测（100 组 Optimal vs RL-Greedy + 最优 SAC） | ✅   |
|                          | 两层系统总体架构图（图1-1）与指针网络机制图（图3-0）         | ✅   |
| **中期之后（论文主体）** | TSP 训练/解码中引入障碍物惩罚或含障代价模型                  | ⏳   |
|                          | 扩展联合评测（500 组），系统分析路径质量与任务成功率         | ⏳   |
|                          | 连续任务 vs 单点成功率落差及障碍物惩罚 TSP 机理解释          | ⏳   |
| **远期展望**             | LLM/Agent 端到端任务交互                                     | ⏳   |

**核心科学问题**：单点避障成功率（95.2%）与整链联合成功率（67–82%）存在显著落差，成功组航程相近则揭示上层几何规划与下层含障执行相互割裂。无人机碰撞存在炸机风险且任务成本较高，**整链成功率较路径几何最优更具工程优先级**。如何通过 TSP 训练引入障碍物惩罚与扩展联合评测，在路径长度较优的基础上提升连续任务整链成功率。

---

## 实验设计总览

| 实验             | 脚本                          | 目的                             | 环境                                       | 训练步数    | 汇总目录                                                   |
| ---------------- | ----------------------------- | -------------------------------- | ------------------------------------------ | ----------- | ---------------------------------------------------------- |
| **上层 TSP**     | `build_dataset.py`            | 500 实例路径/ratio 对比          | obstacle_map（SimpleAvoid 同尺度整数地图） | —           | `PointerNetwork/.../build_dataset_output/`                 |
| **下层算法对比** | `run_expA_train_benchmark.py` | SAC/TD3/PPO + 感知消融           | SimpleAvoid + AirSim                       | 20 万       | `logs/ExpA_benchmark/20260615_182441/`                     |
| **联合评测**     | `joint_dataset_test.py`       | Optimal vs RL-TSP 序列任务成功率 | SimpleAvoid + **最优 SAC**                 | 100 组×1 ep | `logs/SimpleAvoid/2025_12_01_20_57_Multirotor_No_CNN_SAC/` |

**统一下层训练设置（算法对比实验）**：`dynamic_name=Multirotor`，`reward_final`，`accept_radius=2m`，`crash_distance=2m`，深度图 60×90，`batch_size=512`，`action_noise_sigma=0.15`。

**指标来源**：训练 rollout（`training_metrics.csv` → benchmark 曲线）；联合评测为固定数据集确定性 rollout（与训练指标不同）。

---

## 科学问题与贡献（承诺—验证映射）

| 开题承诺                                               | 中期验证状态                                             | 对应论文章节 |
| ------------------------------------------------------ | -------------------------------------------------------- | ------------ |
| 上层：无障碍 TSP 的 RL 指针网络训练与 ratio 等指标评测 | ✅ 500 实例，RL-Greedy $L/L_{\mathrm{opt}}\approx 1.006$ | 第3章        |
| 下层：AirSim 中 RL 导航避障训练与成功率评测            | ✅ 四组算法对比 benchmark + 单点 500 次评测              | 第4章        |
| 上下层初步联合评测（路径质量—任务成功率）              | ✅ 100 组（最优 SAC：`2025_12_01_20_57_...`，见下表）    | 第5章        |
| TSP 训练引入障碍物惩罚、扩展联合评测                   | ⏳ 中期之后（论文主体）                                  | 第3/5章      |
| LLM/Agent 端到端交互                                   | ⏳ 远期                                                  | 第6章展望    |

---

## 章节骨架（预定）

### 第1章 绪论

- 背景：农业巡检、工业运维等多目标 UAV 作业 → `@huang2022ufpmp`, `@zhu2021tph`
- 瓶颈：传统建图规划实时性；序列规划与连续避障割裂 → `@sariff2006overview`, `@taketomi2017vslam`
- 切入点：上层 Pointer Network TSP + 下层 SAC 避障的两层架构 → `@bello2017nco`, `@haarnoja2018sac`
- **系统总览图**：`thesis/midterm/figures/Overall_Architecture.png`（图1-1）
- 论证主线：单点 95.2% vs 整链 67–82%（连续任务损耗）→ 成功组航程相近（上下层割裂）→ 无人机成功率优先于路径最优 → TSP 训练引入障碍物惩罚

### 第2章 相关技术与研究现状

（同前，略）

### 第3章 上层多目标访问序列规划

#### 指针网络 TSP 策略架构（上层主策略）

本课题上层实现于 `PointerNetwork-RL-TSP_pytorch/PointerNetwork/`，遵循 Bello 等人 Neural Combinatorial Optimization（NCO）~\cite{bello2017nco} 第 5.1 节对称 TSP 设定，指针解码继承 Vinyals 等人 Pointer Network~\cite{vinyals2015pointer}（机制示意：`thesis/midterm/figures/pointer_network_architecture.png`）。代码将 NCO 原文的 `[0,1]^2` 均匀采样扩展为 **obstacle_map**（与 SimpleAvoid 同尺度的整数障碍地图，默认 `[-60,60]^2`），便于与下层联合评测对齐；对照实验可切换 `unit_square`。

**数据流（城市坐标 → 访问序列）**

```
DataGenerator (obstacle_map / unit_square)
  → 输入 batch (B, n, 2)，对称 TSP、无 depot
  → Encoder：Conv1d(2→128) + LSTM(128→128) → 节点嵌入 (B, n, 128)
  → PointerDecoder：LSTMCell 逐步解码 n 步
       每步：Glimpse 注意力 → 指针 logits（mask 已访问）→ 采样/贪心选城市
  → 输出排列 π 及 log p(π)；目标 L(π) = 闭合回路欧氏长度 tour_length
  → Critic：同构 Encoder + 3×Glimpse process + FFN → 基线 b(s)
  → Actor-Critic REINFORCE 更新 Actor / Critic
```

**网络结构**（`model.py`）

| 模块                 | 结构                                                        | 说明                                                             |
| -------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------- |
| **Encoder（Actor）** | `Conv1d(2→128, k=1)` + `LSTM(128→128)`                      | 将各城市 $(x,y)$ 嵌入为隐向量序列                                |
| **GlimpseBlock**     | $w_{ref}$ Conv1d + $w_q$ Linear + $v$·tanh                  | 单头注意力 glimpse；Decoder 端带 mask                            |
| **PointerDecoder**   | `LSTMCell(128→128)` + Glimpse + 指针层                      | logits $= C·\tanh(scores)$，$C=10$；推理时 scores/$T$（$T=2.0$） |
| **Critic**           | 同构 embed+LSTM + **3 步** process block + FFN$[128→128→1]$ | 输出标量基线 $b(s)$；末层 bias 初值为 $n/2$                      |
| **目标**             | `tour_length`                                               | 对称 TSP 闭合回路长度（最小化）                                  |

**训练（Actor-Critic REINFORCE，NCO Algorithm 1，`engine.py`）**

| 项目        | 配置                                      | 说明                              |
| ----------- | ----------------------------------------- | --------------------------------- |
| 算法        | REINFORCE + 可学习 Critic 基线            | 非 PPO/A2C；与 NCO 一致           |
| Actor 损失  | $\mathbb{E}[(L(\pi)-b(s))·\log p(\pi)]$   | $L$ 为回路长度，detach 基线与奖励 |
| Critic 损失 | $\mathrm{MSE}(b(s), L(\pi))$              | 独立 Adam 优化 Critic 参数        |
| 优化器      | Adam $\beta_1=0.9,\beta_2=0.999$          | Actor/Critic 共享初始 lr          |
| TSP20 超参  | batch 128，embed/hidden 128，lr $10^{-3}$ | 每 5000 步 ×0.96 衰减             |
| 迭代        | 100k（NCO 原文 250k）                     | `config.py` `--iteration`         |
| 梯度裁剪    | $\|\nabla\|_2 \le 1.0$                    | `grad_clip_norm`                  |
| 输入打乱    | `shuffle_input=True`                      | 训练/解码前随机置换城市顺序       |

**数据生成**（`dataset.py`）

| 模式                   | 说明                                                |
| ---------------------- | --------------------------------------------------- |
| `obstacle_map`（默认） | 在 SimpleAvoid 同布局障碍外采样 $n=20$ 个整数安全点 |
| `unit_square`          | NCO 原文 $[0,1]^2$ 连续均匀，用于复现对照           |

**推理解码（NCO Table 1–2，`build_dataset.py` / `engine.py`）**

| 方法            | decode_strategy                  | 中期 500 实例设置                                      |
| --------------- | -------------------------------- | ------------------------------------------------------ |
| **RL-Greedy**   | `greedy`                         | 逐步 argmax；**默认上层输出**                          |
| **RL-Sampling** | `best_of_k`                      | balanced 预设：**256** 次采样取最短                    |
| **RL-AS**       | `pretrained_active_search`       | 300 步实例级微调，EMA 基线 $\alpha=0.99$，lr $10^{-5}$ |
| 基线            | Christofides / 2-opt / Held-Karp | `build_dataset.py` 同批对比                            |

**代码映射**

| 文件               | 职责                                                     |
| ------------------ | -------------------------------------------------------- |
| `model.py`         | `PointerNetwork` = Encoder + `PointerDecoder` + `Critic` |
| `engine.py`        | `train_loop`、`decode_tour`、`active_search`             |
| `config.py`        | NCO TSP20 超参预设                                       |
| `dataset.py`       | `DataGenerator`                                          |
| `build_dataset.py` | 500 实例批量评测 → `dataset_500.npy` / CSV               |

- **中期结果（500 实例，obstacle_map，eval_preset=balanced）**：
  - Optimal 405.65；RL-Greedy 408.18（ratio **1.006**）
  - RL-Sampling/RL-AS ratio **≈1.001**；267/500 RL-Greedy 达最优
- **中期之后**：TSP 训练/解码引入障碍物惩罚，在保持 ratio 可接受的前提下提升联合评测成功率
- **图表**：`paper_results/fig_ratio_curve.png`，`summary_table.csv`
- **文献 PDF**：`references/NEURAL COMBINATORIAL.pdf`（Bello ICLR 2017）

### 第4章 下层 DRL 导航避障

#### SAC+DepthMaxPool 策略架构（下层主策略）

本课题下层固定采用 **SAC + DepthMaxPool 特征提取器**（`scripts/utils/custom_policy_sb3.py`）：对 AirSim 深度图做 **MaxPool 区域池化**（无卷积层），将 $60\times90$ 视野压缩为 12 维区域最大深度，再与 2 维状态拼接，计算量小、训练稳定。环境封装与 MDP 建模继承 He 等人 AirSim DRL 导航框架~\cite{he2021explainable}（原文 TD3+CNN-GAP；本文改为 SAC+DepthMaxPool，并扩展上层 TSP 联合评测）。

**数据流（AirSim → 策略网络）**

```
AirSim DepthVis
  → resize 60×90，裁剪至 [0,15] m 并反相映射为 0–255
  → 与状态特征拼成双通道观测 (60, 90, 2)
  → SB3 归一化至 [0,1]
  → DepthMaxPool：MaxPool(16×20) → 12 维深度特征 + 2 维状态 → 14 维
  → Actor/Critic 共享 MLP [64, 32, 16]（tanh）
  → Actor 输出 Squashed Gaussian 动作 (v_xy, yaw_rate)
```

**环境观测与动作**（`gym_env/gym_env/envs/airsim_env.py`，2D 多旋翼）

| 项目             | 配置                                      | 说明                                              |
| ---------------- | ----------------------------------------- | ------------------------------------------------- |
| 感知             | `perception=depth`                        | AirSim 深度图 + 状态嵌入第二通道                  |
| 观测空间         | `Box(60, 90, 2)` uint8                    | 通道 0：深度图；通道 1 第 0 行：状态向量          |
| 深度图           | 60×90，max 15 m                           | 近处高亮（255−归一化深度）                        |
| 状态特征（2 维） | `distance_norm`, `relative_yaw_norm`      | 2D 导航、不含速度状态                             |
| 动作空间         | 2 维连续                                  | $v_{xy}\in[0.5,5]$ m/s，$\dot\psi\in[-30°,30°]/$s |
| 控制模型         | `MultirotorDynamicsSimple`                | $dt=0.1$ s，速度级指令                            |
| 奖励             | `reward_final`                            | 距离塑形 + 姿态/障碍/动作惩罚；成功 +10，碰撞 −20 |
| 终止             | `accept_radius=2` m，`crash_distance=2` m | 到达或碰撞结束 episode                            |

**DepthMaxPool 特征提取**（`custom_policy_sb3.DepthMaxPool`）

| 模块     | 输入                        | 操作                         | 输出维                    |
| -------- | --------------------------- | ---------------------------- | ------------------------- |
| 深度分支 | 通道 0，$1\times60\times90$ | `MaxPool2d(16,20)` + Flatten | **12**（$3\times4$ 网格） |
| 状态分支 | 通道 1 第 0 行前 2 元素     | 直接读取                     | **2**                     |
| 拼接     | —                           | `concat`                     | **14** = `features_dim`   |

**SAC 策略网络**（SB3 `sac/policies.py`，`thread_train.py` 实例化）

| 组件        | 结构                                      | 说明                                                   |
| ----------- | ----------------------------------------- | ------------------------------------------------------ |
| Policy 基类 | `CnnPolicy`                               | 接入自定义 `features_extractor_class=DepthMaxPool`     |
| Actor       | 14 → MLP [64,32,16] → $\mu$, $\log\sigma$ | `SquashedDiagGaussianDistribution`，2 维动作           |
| Critic      | Twin Q，共享特征提取器                    | $Q_1, Q_2$：$[\text{feat};\text{action}]$ → MLP → 标量 |
| 探索        | `NormalActionNoise`，$\sigma=0.15$        | 数据采集阶段动作噪声                                   |
| 训练        | off-policy replay buffer                  | 见下表                                                 |

**训练超参**（算法对比实验统一配置；最优模型 $10^5$ 步）

| 参数                            | 值                                            |
| ------------------------------- | --------------------------------------------- |
| `total_timesteps`               | $2\times10^5$（对比实验）/ $10^5$（最优 SAC） |
| `learning_rate`                 | $10^{-3}$                                     |
| `gamma`                         | 0.99                                          |
| `buffer_size`                   | 50000                                         |
| `batch_size`                    | 512                                           |
| `learning_starts`               | 2000                                          |
| `train_freq` / `gradient_steps` | 100 / 100                                     |
| `net_arch`                      | [64, 32, 16]                                  |
| `activation_function`           | tanh                                          |
| `cnn_feature_num`               | 12                                            |

**代码映射**

| 功能                  | 路径                                                                           |
| --------------------- | ------------------------------------------------------------------------------ |
| 环境封装              | `gym_env/gym_env/envs/airsim_env.py`                                           |
| 动力学                | `gym_env/gym_env/envs/dynamics/multirotor_simple.py`                           |
| DepthMaxPool 特征提取 | `scripts/utils/custom_policy_sb3.py`                                           |
| 训练入口              | `scripts/utils/thread_train.py`                                                |
| 对比实验配置          | `configs/config_ExpA_SAC_NoCNN.ini`（历史文件名，内容为 DepthMaxPool）         |
| SB3 SAC 算法          | `stable-baselines3/stable_baselines3/sac/`                                     |
| 最优 checkpoint       | `logs/SimpleAvoid/2025_12_01_20_57_Multirotor_No_CNN_SAC/models/model_sb3.zip` |

**设计要点**：深度图保留空间结构，MaxPool 以 $16\times20$ 步长覆盖 60×90 视野，得到 12 个区域最大深度，等价于粗粒度障碍占用栅格；状态特征嵌入第二通道避免额外 MLP 输入分支，与 SB3 `CnnPolicy` 接口一致。相较 SAC+mlp（5 维手工向量特征），DepthMaxPool 利用完整深度图，对比实验成功率 94% vs 61%。

**与基线文献差异**（`he2021explainable` / `uav-nav.pdf`）

| 项目     | He 等 (2021)                  | 本文                                      |
| -------- | ----------------------------- | ----------------------------------------- |
| 算法     | TD3                           | **SAC**（对比 TD3/PPO）                   |
| 感知网络 | CNN + GAP                     | **DepthMaxPool**（MaxPool 12 维，无卷积） |
| 维度     | 3D（$v_{xy}, v_z, \dot\psi$） | 2D SimpleAvoid（$v_{xy}, \dot\psi$）      |
| 任务     | 单目标 reactive 导航          | 单目标 + **上层 TSP 多目标联合评测**      |
| 可解释性 | SHAP-CAM 可视化               | 未作为中期重点（可选后续）                |

#### 实验 A：算法与感知对比（4 组，20 万步）

数据源：`scripts/logs/ExpA_benchmark/20260615_182441/benchmark_summary.csv`

| 实验             | 算法 | 策略/感知                          | 最终成功率 | 峰值成功率 | 最终 reward | 撞毁率 |
| ---------------- | ---- | ---------------------------------- | ---------- | ---------- | ----------- | ------ |
| SAC+DepthMaxPool | SAC  | depth + DepthMaxPool (MaxPool 12d) | **94%**    | **98%**    | +9.55       | 6%     |
| TD3+DepthMaxPool | TD3  | depth + DepthMaxPool               | 75%        | 84%        | +3.72       | 24%    |
| SAC+mlp          | SAC  | vector + MlpPolicy                 | 61%        | 74%        | −13.95      | 37%    |
| PPO+DepthMaxPool | PPO  | depth + DepthMaxPool               | 0%         | 0%         | −244.24     | 5%     |

**解读**：同环境同步数下 SAC+DepthMaxPool 显著最优；PPO 未收敛（缺 `[PPO]` 专用超参）；纯向量 mlp 弱于深度图 MaxPool。

**图表**：`benchmark_success_rate.png`，`benchmark_crash_rate.png`，`benchmark_ep_rew_mean.png`，`benchmark_ep_len_mean.png`（中期报告图4-1 四联图）

**对照训练目录**：`2026_06_11_17_06_...` 等（算法/感知横向 benchmark，rollout 指标）。

**最优 SAC（联合评测下层策略）**：`logs/SimpleAvoid/2025_12_01_20_57_Multirotor_No_CNN_SAC/`  
SAC + DepthMaxPool，$10^5$ 步，SimpleAvoid，`reward_final`，depth + MaxPool 12 维。

#### 单点导航评测（500 次，2026-06-16）

数据源：`single_point_eval_500_training_20260616_172747.csv`

| 指标           | 结果                 |
| -------------- | -------------------- |
| 起点           | $(0,0,5)$ m          |
| 目标采样       | training 模式        |
| 成功率         | **95.2%**（476/500） |
| 碰撞率         | 4.8%（24/500）       |
| 成功时平均步数 | 228.40（σ=28.06）    |
| 成功时平均航程 | 54.38 m（σ=3.79）    |

### 第5章 上下层联合评测

- 数据：`dataset_500.npy` 中 **optimal_coords** 与 **tsp_coords** 两组对齐序列（100 组）
- 流程：`joint_dataset_test.py` + 最优 SAC 逐点导航
- 评测记录：`run_id=20260610_211452`（2026-06-10）
- 模型：`.../2025_12_01_20_57_Multirotor_No_CNN_SAC/models/model_sb3.zip`

| 规划方法                               | 成功率  | 碰撞率  | 成功/总数 | 成功时平均航程/m | 标准差/m |
| -------------------------------------- | ------- | ------- | --------- | ---------------- | -------- |
| Optimal（Held-Karp，`optimal_coords`） | 67%     | 24%     | 67/100    | 499.86           | 35.32    |
| RL-Greedy（指针网络，`tsp_coords`）    | **82%** | **11%** | 82/100    | 497.44           | 39.82    |

**解读**：联合评测仅对比 `dataset_500.npy` 中两组对齐序列。RL-Greedy 整链成功率（82%）高于几何最优（67%），但二者均明显低于单点导航（95.2%），说明**连续逐点任务显著拉低避障成功率**。成功组平均航程相近（497.44 m vs 499.86 m），说明**上层几何序列规划与下层连续避障优化相互割裂**——上层仅最短化无障碍回路，下层才面对真实障碍；差异主要体现在能否完成与碰撞中断（24% vs 11%），而非成功时飞得更远。鉴于无人机碰撞存在炸机风险且任务成本较高，**整链成功率较路径几何最优更具优先级**，可适度牺牲路径最优性换取成功率。中期之后在 TSP 训练/解码中引入障碍物惩罚，使上层规划与下层可飞性对齐，在路径长度较优的基础上提升连续任务整链成功率。

**待办（中期之后）**：TSP 训练/解码加入障碍惩罚；联合评测扩展至 500 组；补充 Optimal 失败 vs RL-Greedy 成功对比轨迹图；量化路径 ratio 与联合成功率的协同关系。

### 第6章 总结与展望

- 已完成：上层 TSP、下层算法对比 benchmark、单点导航评测、100 组初步联合评测（最优 SAC）
- 中期之后：障碍物惩罚 TSP、500 组联合评测、路径质量—成功率协同分析
- 远期：PPO 重训、LLM/Agent 交互

---

## 图表资产索引

| 编号建议 | 文件                                                      | 用途                                          |
| -------- | --------------------------------------------------------- | --------------------------------------------- |
| 图1-1    | `thesis/midterm/figures/Overall_Architecture.png`         | 两层系统总体架构（多目标任务→TSP→SAC→轨迹）   |
| 图3-0    | `thesis/midterm/figures/pointer_network_architecture.png` | 指针网络 Encoder-Decoder 与 pointing 机制     |
| 图3-1    | `paper_results/fig_ratio_curve.png`                       | TSP ratio 曲线                                |
| 表3-1    | `mean_length_summary.csv`                                 | TSP 路径长度                                  |
| 图4-0    | `thesis/midterm/figures/SAC_DepthMaxPool.png`             | SAC+DepthMaxPool 三层架构（感知→MaxPool→SAC） |
| 图4-1    | `thesis/midterm/figures/fig_expA_*.png`（四联）           | 下层 rollout：成功率/碰撞率/奖励/步数         |
| 表4-1    | `ExpA_benchmark/.../benchmark_summary.csv`                | 算法对比数值                                  |
| 表4-2    | —（见中期报告 `tab:sac-arch`）                            | SAC+DepthMaxPool 架构                         |
| 表4-3    | —（见中期报告 `tab:sac-hparam`）                          | SAC 训练超参                                  |
| 表4-4    | `single_point_eval_500_training_*.csv`                    | 单点导航 500 次评测                           |
| 表5-1    | `2025_12_01_.../compare_optimal_vs_tsp.csv`               | 联合评测（最优 SAC）                          |
| 图5-1    | `joint_dataset_test_example/tsp_route_loop.png`           | 上层规划路径（RL-Greedy）                     |
| 图5-2    | `joint_dataset_test_example/uav_route_loop.png`           | 下层 SAC 实际轨迹俯视图                       |
| 图5-3    | `joint_dataset_test_example/uav.png`                      | AirSim 含障环境实时仿真                       |

---

## 系统架构（代码映射）

```
┌─────────────────────────────────────────────────────────┐
│  上层：PointerNetwork-RL-TSP_pytorch                     │
│  build_dataset.py → dataset_500.npy                      │
└──────────────────────────┬──────────────────────────────┘
                           │ optimal_coords / tsp_coords
┌──────────────────────────▼──────────────────────────────┐
│  下层：gym_env/airsim_env.py + SB3 (SAC/TD3/PPO)         │
│  最优 SAC：2025_12_01_20_57_...（联合评测下层）          │
│  joint_dataset_test.py → 联合评测                         │
└─────────────────────────────────────────────────────────┘
```

| 模块         | 路径                                                 |
| ------------ | ---------------------------------------------------- |
| TSP          | `PointerNetwork-RL-TSP_pytorch/PointerNetwork/`      |
| 环境         | `gym_env/gym_env/envs/airsim_env.py`                 |
| 基线文献     | `references/uav-nav.pdf` → `@he2021explainable`      |
| 动力学       | `gym_env/gym_env/envs/dynamics/multirotor_simple.py` |
| DepthMaxPool | `scripts/utils/custom_policy_sb3.py`                 |
| 训练         | `scripts/utils/thread_train.py`                      |
| SB3 SAC      | `stable-baselines3/stable_baselines3/sac/`           |
| 算法对比     | `scripts/run_expA_train_benchmark.py`                |
| 联合评测     | `scripts/joint_dataset_test.py`                      |

---

## 术语一致性

| 中文                           | 英文                                                  | 章节 |
| ------------------------------ | ----------------------------------------------------- | ---- |
| 指针网络                       | Pointer Network                                       | 2/3  |
| 旅行商问题                     | TSP                                                   | 1/3  |
| 软演员-评论家                  | SAC                                                   | 2/4  |
| 最优性比率                     | $L/L_{\mathrm{opt}}$                                  | 3    |
| 联合评测                       | joint evaluation                                      | 5    |
| 路径质量—任务成功率协同        | path quality vs task success synergy                  | 1/5  |
| 上下层规划—执行割裂 | upper–lower planning–execution decoupling | 1/5  |
| 整链成功率优先于路径最优 | chain success over geometric optimality | 1/5  |
| 障碍物惩罚 TSP                 | obstacle-penalized TSP                                | 3/5  |
| DepthMaxPool                   | 深度 MaxPool 编码器（depth + regional MaxPool）       | 4    |
| No_CNN                         | DepthMaxPool 历史别名（勿改旧 checkpoint 路径）       | 4    |
| 特征维                         | 14 = 12 (depth) + 2 (state)                           | 4    |

---

## 文档索引

- 引用注册表：[`CITATION_REGISTRY.md`](CITATION_REGISTRY.md)
- BibTeX：[`references.bib`](references.bib)
- 中期 LaTeX：[`midterm/report.tex`](midterm/report.tex)
- 中期模板结构：[`midterm_template_structure.md`](midterm_template_structure.md)
