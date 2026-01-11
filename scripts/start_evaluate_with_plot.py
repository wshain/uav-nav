import sys
import argparse
import math
import numpy as np
import os
from PyQt5 import QtWidgets

# 自定义模块导入
from utils.thread_evaluation import EvaluateThread  # 用于在后台线程中运行模型评估
from utils.ui_train import TrainingUi               # GUI 界面类
from configparser import ConfigParser              # 用于读取 .ini 配置文件


def get_parser():
    """
    定义命令行参数解析器（当前未在 main() 中使用，但保留以支持未来扩展）。
    """
    parser = argparse.ArgumentParser(
        description="trained model evaluation with plot")
    parser.add_argument('-model_path', required=True, 
                        help='要评估的模型路径，只需复制日志文件夹的相对路径')
    parser.add_argument('-eval_eps', required=True, type=int,
                        help='评估的 episode 数量')
    return parser


def main():
    """
    主函数：启动带图形界面的 DRL 模型评估程序。
    加载指定路径下的训练模型与配置，在 AirSim 环境中运行评估，
    并将动作、状态、轨迹、奖励等数据实时绘制到 PyQt5 GUI 上。
    """

    # ────────────────────────────────────────
    # 1. 设置评估所需路径（硬编码，便于调试）
    # ────────────────────────────────────────
    # 指定训练日志根目录（包含 config/ 和 models/ 子文件夹）
    eval_path = r'C:/Users/22864/UAV_Navigation_DRL_AirSim/scripts/logs/SimpleAvoid/2025_12_01_20_57_Multirotor_No_CNN_SAC'

    # 构造配置文件和模型文件的完整路径
    config_file = eval_path + '/config/config.ini'          # 配置文件路径
    model_file = eval_path + '/models/model_sb3.zip'       # Stable-Baselines3 模型文件
    total_eval_episodes = 50                               # 评估轮数（episodes）

    # config_file = r"D:\OneDrive - mail.nwpu.edu.cn\Github\PhD-thesis-plot\...\config.ini"
    # model_file = eval_path + '/models/model_200000.zip'

    # ────────────────────────────────────────
    # 2. 初始化 PyQt5 图形界面
    # ────────────────────────────────────────
    app = QtWidgets.QApplication(sys.argv)   # 创建 Qt 应用实例（必须在 GUI 前创建）
    gui = TrainingUi(config=config_file)     # 初始化自定义 GUI 窗口，传入配置用于初始化图表
    gui.show()                               # 显示主窗口

    # ────────────────────────────────────────
    # 3. 创建并启动评估线程（后台运行推理，避免阻塞 GUI）
    # ────────────────────────────────────────
    
    # 可选：定义连续目标点序列（用于连续测试模式）
    # 如果不设置 goal_sequence，则使用传统的每次重置环境的方式
    # 连续目标点模式：定义一系列目标点，无人机将依次到达这些目标点，不重置环境
    
    # SimpleAvoid 环境工作空间：x: [-60, 60], y: [-60, 60], z: [0.5, 50]
    # 起始位置：[0, 0, 5]
    # 
    # 障碍物位置（根据地图）：
    # - 中心区域方形障碍物：(-10, -10), (10, -10), (-10, 10), (10, 10)
    # - 上方方形障碍物：(0, 30)
    # - 四个角落圆形障碍物：(-30, -30), (30, -30), (-30, 30), (30, 30)
    #
    # 设计策略：避开这些区域，使用更大的安全距离
    # 1. 避开中心区域 (±10, ±10) - 使用距离 > 20米
    # 2. 避开上方 (0, 30) - 向北时使用 y < 25 或 y > 35
    # 3. 避开四个角落 (±30, ±30) - 使用距离 > 35米，远离角落
    # 4. 使用外层距离（35-45米），确保与所有障碍物保持足够距离
    
    # 注意：SimpleAvoid 工作空间是 x: [-60, 60], y: [-60, 60]
    # 目标点必须在工作空间内，否则会触发 is_not_in_workspace 导致提前结束
    # 所有目标点坐标必须在 [-60, 60] 范围内
    
    # 从 TSP main.py 生成的坐标文件中加载目标点序列
    tsp_goal_file = r'C:/Users/22864/UAV_Navigation_DRL_AirSim/PointerNetwork-RL-TSP/PointerNetwork/save/goal_sequence.npy'
    
    if os.path.exists(tsp_goal_file):
        try:
            goal_sequence_array = np.load(tsp_goal_file)
            # 转换为列表格式，每个元素是 [x, y, z]
            goal_sequence = goal_sequence_array.tolist()
            print(f"成功加载 TSP 生成的坐标序列，共 {len(goal_sequence)} 个目标点")
            print(f"目标点序列: {goal_sequence}")
        except Exception as e:
            print(f"加载 TSP 坐标文件失败: {e}")
            print("使用默认目标点序列")
            goal_sequence = [
                [-40, -40, 5],       
                [40, -40, 5],     
                [40, 40, 5],    
                [-20, 40, 5]
            ]
    else:
        print(f"TSP 坐标文件不存在: {tsp_goal_file}")
        print("使用默认目标点序列")
        goal_sequence = [
            [-40, -40, 5],       
            [40, -40, 5],     
            [40, 40, 5],    
            [-20, 40, 5]
        ]
    
    # 如果不使用连续目标点模式，取消下面的注释，并注释掉上面的 goal_sequence 加载代码
    # goal_sequence = None  # 设置为 None 使用传统模式（每次测试后重置环境）
    
    evaluate_thread = EvaluateThread(
        eval_path=eval_path,
        config=config_file,
        model_file=model_file,
        eval_ep_num=total_eval_episodes,
        goal_sequence=goal_sequence  # 传入目标点序列（可选）
    )

    # 将环境（env）发出的信号连接到 GUI 的回调函数，实现数据实时更新
    evaluate_thread.env.action_signal.connect(gui.action_cb)        # 动作数据 → 动作显示
    evaluate_thread.env.state_signal.connect(gui.state_cb)          # 状态数据 → 状态表格
    evaluate_thread.env.attitude_signal.connect(gui.attitude_plot_cb)  # 姿态 → 姿态曲线图
    evaluate_thread.env.reward_signal.connect(gui.reward_plot_cb)    # 奖励 → 奖励曲线图
    evaluate_thread.env.pose_signal.connect(gui.traj_plot_cb)       # 位置 → 3D 轨迹图

    # ────────────────────────────────────────
    # 4. 可选：如果启用了 LGMD 仿生视觉模块，则连接其专用信号
    # ────────────────────────────────────────
    cfg = ConfigParser()
    cfg.read(config_file)
    if cfg.has_option('options', 'perception'):
        if cfg.get('options', 'perception') == 'lgmd':
            evaluate_thread.env.lgmd_signal.connect(gui.lgmd_plot_cb)  # LGMD 输出 → 专用绘图

    # 启动评估线程（调用其 run() 方法，在子线程中执行）
    evaluate_thread.start()

    # ────────────────────────────────────────
    # 5. 进入 Qt 事件循环，保持 GUI 响应直到用户关闭窗口
    # ────────────────────────────────────────
    # 注意：sys.exit(app.exec_()) 会阻塞主线程并接管程序控制流，
    #      因此其后的 print 语句永远不会被执行。
    sys.exit(app.exec_())


# ────────────────────────────────────────
# 程序入口点
# ────────────────────────────────────────
if __name__ == "__main__":
    main()  # 启动带 GUI 的评估程序