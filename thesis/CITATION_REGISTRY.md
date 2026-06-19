# 引用注册表（Citation Registry）

> **维护规则**：每次在论文/报告中 `\cite{}` 一条文献，必须在本表追加或更新一行，并填写「引用理由」与「使用位置」。BibTeX 键名与 `references.bib` 保持一致。

---

## 引用总表

| BibTeX Key | 文献简述 | 引用理由 | 使用位置 |
|------------|----------|----------|----------|
| `huang2022ufpmp` | UFPMP-Det 无人机目标检测 | 支撑绪论中“无人机在复杂场景感知需求”的背景 | 开题报告§1；中期§1；正文第1章 |
| `zhu2021tph` | TPH-YOLOv5 航拍检测 | 同上，说明航拍场景目标检测代表性工作 | 开题报告§1；正文第1章 |
| `liu2024navagent` | NavAgent 城市 VLN | 说明 UAV 具身导航与多模态感知研究趋势 | 开题报告§1；正文第2章 |
| `sutton2018rl` | Sutton RL 导论 | DRL 理论基础，界定 MDP 与学习范式 | 开题报告§2；正文第2章 |
| `mnih2015dqn` | Nature DQN | 深度 RL 里程碑，对比本文 continuous control | 开题报告§2；正文第2章 |
| `seo2024nature` | Nature DRL 应用 | 说明 DRL 在复杂控制中的有效性（背景） | 开题报告§2 |
| `adiuku2024navyolo` | NAV-YOLO 移动机器人避障 | 传统感知+规划模块化方案代表 | 开题报告§2.1；正文第2章 |
| `vemprala2024chatgpt` | ChatGPT for Robotics | LLM 与机器人结合的设计原则，支撑“远期 Agent”展望 | 开题报告§2.1；正文第2/6章 |
| `wang2024parallel` | 平行驾驶与大模型 | 具身智能与大模型融合背景 | 开题报告§2.1 |
| `zhang2024vln` | VLN 综述 | VLN 领域全景，定位本文与 VLN 的差异 | 开题报告§2.1；正文第2章 |
| `zhou2024navgpt` | NavGPT | LLM 零样本 VLN 代表，说明纯 LLM 定量指标不足 | 开题报告§2.1；正文第2章 |
| `macenski2020nav2` | Navigation2 | 模块化 ROS 导航框架基线 | 开题报告§2.1；正文第2章 |
| `he2021explainable` | He 等可解释 DRL UAV 导航 | **本文 AirSim/Gym 环境与 DRL 导航代码基线**；深度图+状态、MDP 建模、reward 设计 | 中期§2；正文第2/4章 |
| `chen2022uavsim` | 端到端 UAV 仿真平台 | SLAM+规划集成仿真平台参考（非本文代码直接来源） | 开题报告§2.2；正文第2章 |
| `zhang2024uninavid` | Uni-NaVid VLA | 国内 VLA 导航代表，对比本文非语义导航路线 | 开题报告§2.2；正文第2章 |
| `fang2024sac` | 安全 SAC 无人机路径规划 | 安全 RL / CMDP 建模与 SAC 应用参考 | 开题报告§2.2；正文第2/4章 |
| `liu2022hrl` | 层次注意力 DRL UAV 导航 | UAV DRL 导航代表性工作，对比本文两层架构 | 开题报告§2.2；正文第2/4章 |
| `dewangan2019gwo` | 3D 路径规划 GWO | 论证 TSP/路径规划 NP-hard 与优化类方法 | 开题报告§2.3；正文第2/3章 |
| `sariff2006overview` | 路径规划算法综述 | A*、Dijkstra、RRT 等传统方法概述 | 开题报告§2.3；正文第2章 |
| `taketomi2017vslam` | Visual SLAM 综述 | SLAM 依赖与实时性瓶颈 | 开题报告§2.3；正文第2章 |
| `wei2022cot` | Chain-of-Thought | LLM 推理链背景，支撑 VLN/LLM 小节 | 开题报告§2.3；正文第2章 |
| `anderson2018vln` | R2R VLN 基准 | VLN 经典数据集与任务定义 | 开题报告§2.3；正文第2章 |
| `brockman2016gym` | OpenAI Gym | RL 环境接口标准，类比 gym_env 设计 | 开题报告§2.3；正文第4章 |
| `haarnoja2018sac` | SAC 算法 | **本文下层导航核心算法** | 开题报告§2.3；中期§2；正文第2/4章 |
| `haarnoja2017sac` | 能量基策略 RL | SAC 前身，算法谱系 | 开题报告§2.3 |
| `shalev2016safe` | 安全多智能体 RL | 安全 RL 背景 | 开题报告§2.3 |
| `ding2023glop` | 安全多智能体 GL 优化 | 约束 MDP 求解参考 | 开题报告§2.3 |
| `altman2021cmdp` | CMDP 专著 | 约束 MDP 理论定义 | 开题报告§2.3；正文第2章 |
| `yang2021wcsac` | WCSAC | 安全 SAC 变体 | 开题报告§2.3 |
| `maruyama2016ros2` | ROS2 性能 | 机器人中间件背景 | 开题报告§2.3 |
| `koenig2004gazebo` | Gazebo 仿真 | 仿真平台生态 | 开题报告§2.3 |
| `meier2015px4` | PX4 飞控 | 无人机控制栈背景 | 开题报告§2.3 |
| `amsters2019turtlebot` | TurtleBot3 教育平台 | 地面机器人仿真实验参考 | 开题报告§2.3 |
| `fan2020dqn` | DQN 理论分析 | DRL 收敛性讨论 | 开题报告§2.3 |
| `fedus2020replay` | Experience Replay | 经验回放机制 | 开题报告§2.3 |
| `vinyals2015pointer` | Pointer Networks | **上层指针网络结构来源** | 中期§2；正文第3章 |
| `bello2017nco` | Neural Combinatorial Optimization | **上层 RL 训练 TSP 的方法论来源** | 中期§2；正文第3章 |
| `christofides1976` | Christofides TSP 近似 | 经典 TSP 近似算法基线 | 中期§2；正文第3章 |
| `lin1973karp` | Karp 大尺度 TSP | Held-Karp 最优求解参考 | 正文第3章 |
| `hopfield1985soma` | Hopfield TSP 早期 NN | TSP 神经网络求解历史 | 正文第2章（可选） |

---

## 中期报告新增引用说明

以下键名在中期报告中首次正式 `\cite`，理由如下：

### `bello2017nco`
- **理由**：本文上层训练流程（REINFORCE + 指针网络 + greedy/sampling 解码）直接遵循 Bello 等人的 NCO 框架；中期需说明方法来源。
- **位置**：中期报告 §2.1 上层规划方法；正文第3章指针网络架构
- **本地 PDF**：`references/NEURAL COMBINATORIAL.pdf`（ICLR 2017，arXiv:1611.09940）
- **代码映射**：`PointerNetwork-RL-TSP_pytorch/PointerNetwork/{model,engine,config}.py`

### `vinyals2015pointer`
- **理由**：指针机制（attention 指向输入节点）是上层网络结构的核心，需引用原始 Pointer Network 论文。
- **位置**：中期报告 §2.1

### `haarnoja2018sac`
- **理由**：下层导航采用 SB3-SAC，需引用 SAC 原文说明算法选择依据。
- **位置**：中期报告 §2.2 下层导航

### `he2021explainable`
- **理由**：本仓库 `gym_env/airsim_env.py`、深度图+状态观测、Gym 接口与 reward 设计继承自 He 等人的 AirSim DRL 导航框架；原文使用 TD3，本文改用 SAC+DepthMaxPool 并扩展上层 TSP。
- **位置**：中期报告 §2.2；正文第4章
- **本地 PDF**：`references/uav-nav.pdf`

### `chen2022uavsim`
- **理由**：Chen 等人面向 SLAM 的端到端 UAV 仿真平台，可作为相关仿真工作对照；**不宜替代 He 2021 作为本文代码基线引用**。
- **位置**：正文第2章（相关 work，可选）

### `liu2022hrl`
- **理由**：作为 UAV DRL 导航领域对照工作，说明本文下层设计与现有研究的异同。
- **位置**：中期报告 §2.2

### `fang2024sac`
- **理由**：开题报告已论证安全 RL 路径规划，中期下层实验沿用 SAC，引用国内同类安全工作增强连续性。
- **位置**：中期报告 §2.2

### `dewangan2019gwo`
- **理由**：说明 TSP/路径规划问题的 NP-hard 性质，解释上层采用学习型近似算法的动机。
- **位置**：中期报告 §2.1

---

## 待补充文献（后期工作）

| 主题 | 本地 PDF | BibTeX Key |
|------|----------|------------|
| AirSim DRL 导航基线 | `references/uav-nav.pdf` | `he2021explainable` ✅ |
| TSP with obstacles | `references/tsp with obstale.pdf` | `tsp_obstacle_tbd` |
| 障碍物 TSP 联合规划 | 待读 | `joint_tsp_nav_tbd` |

> 读取 PDF 后须补全元数据并更新本表。
