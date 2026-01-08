# ────────────────────────────────────────────────
# 依赖库导入
# ────────────────────────────────────────────────
from PyQt5 import QtCore                    # Qt 多线程支持
from configparser import ConfigParser      # 读取 .ini 配置文件
from stable_baselines3 import TD3, SAC, PPO  # 支持的 DRL 算法
import numpy as np
import gym_env                              # 自定义 AirSim Gym 环境（需提前注册）
import gym
import math
import os
import sys
import cv2                                  # 注意：此处可能无实际用途
from tqdm import tqdm                       # 进度条显示

# 添加项目路径到 Python 模块搜索路径（便于导入自定义模块）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
sys.path.append(r"C:\Users\helei\Documents\GitHub\UAV_Navigation_DRL_AirSim\scripts")


# ────────────────────────────────────────────────
# 规则策略（用于与 DRL 对比）
# ────────────────────────────────────────────────
def rule_based_policy(obs):
    '''
    自定义线性规则策略，用于与 DRL 模型对比（例如在 LGMD 实验中）。
    输入观测值，输出一个受限的偏航角动作（单位：弧度）。
    '''
    # 移除多余维度（假设 obs 形状为 (1, N)）
    obs = np.squeeze(obs, axis=0)

    # 将原始观测 [-1, 1] 映射到 [0, 1]
    for i in range(5):
        obs[i] = obs[i] / 2 + 0.5

    # 定义各方向传感器的权重（模拟人工避障逻辑）
    obs_weight = np.array([1.0, 3.0, 3.0, -3.0, -1.0, 3.0])
    action = obs * obs_weight
    action_sum = np.sum(action)

    # 动作裁剪：限制在 ±40 度（转换为弧度）
    max_angle = math.radians(40)
    if action_sum > max_angle:
        action_sum = max_angle
    elif action_sum < -max_angle:
        action_sum = -max_angle

    return np.array([action_sum])  # 返回形状为 (1,) 的动作


# ────────────────────────────────────────────────
# 评估线程类：在子线程中运行模型推理，避免阻塞 GUI
# ────────────────────────────────────────────────
class EvaluateThread(QtCore.QThread):
    """
    继承自 QThread，用于在后台执行模型评估。
    支持 DRL 模型（TD3/SAC/PPO）或规则策略，并可记录轨迹、状态、动作等数据。
    
    支持两种测试模式：
    1. 传统模式（goal_sequence=None）：每次测试从初始点到达目标点，成功后重置环境
    2. 连续目标点模式（goal_sequence不为None）：有一系列目标点，从一个目标点到达下一个目标点，不重置环境
    
    使用示例：
        # 传统模式
        evaluate_thread = EvaluateThread(
            eval_path=eval_path,
            config=config_file,
            model_file=model_file,
            eval_ep_num=50
        )
        
        # 连续目标点模式
        goal_sequence = [
            [50, 50, 5],      # 第一个目标点 [x, y, z]
            [100, 100, 5],    # 第二个目标点
            [150, 50, 5],     # 第三个目标点
        ]
        evaluate_thread = EvaluateThread(
            eval_path=eval_path,
            config=config_file,
            model_file=model_file,
            eval_ep_num=50,
            goal_sequence=goal_sequence
        )
    """

    def __init__(self, eval_path, config, model_file, eval_ep_num, eval_env=None, eval_dynamics=None, goal_sequence=None):
        super(EvaluateThread, self).__init__()
        print("Initializing evaluation thread...")

        # 1. 读取并可选修改配置文件
        self.cfg = ConfigParser()
        self.cfg.read(config)

        # 若指定了评估环境或动力学模型，则覆盖配置
        if eval_env is not None:
            self.cfg.set('options', 'env_name', eval_env)
            # 特殊处理 NH_center 环境：缩小成功判定半径
            if eval_env == 'NH_center':
                self.cfg.set('environment', 'accept_radius', str(1))

        if eval_dynamics is not None:
            self.cfg.set('options', 'dynamic_name', eval_dynamics)

        # 2. 创建 Gym 环境并注入配置
        self.env = gym.make('airsim-env-v0')
        self.env.set_config(self.cfg)
        
        # 获取原始环境实例（解包所有包装器）
        # 因为环境可能被 Monitor、DummyVecEnv、VecTransposeImage 等包装
        self.raw_env = self.env
        # 如果是 VecEnv（如 DummyVecEnv），获取第一个环境
        if hasattr(self.env, 'envs') and len(self.env.envs) > 0:
            self.raw_env = self.env.envs[0]
        # 继续解包，直到找到 AirsimGymEnv
        # 需要处理 VecEnvWrapper（使用 venv 属性）和 gym.Wrapper（使用 env 属性）
        while True:
            # 检查是否是 AirsimGymEnv
            if hasattr(self.raw_env, '__class__') and 'AirsimGymEnv' in str(self.raw_env.__class__):
                break
            # 尝试解包 VecEnvWrapper
            if hasattr(self.raw_env, 'venv'):
                self.raw_env = self.raw_env.venv
                continue
            # 尝试解包 gym.Wrapper
            if hasattr(self.raw_env, 'env'):
                self.raw_env = self.raw_env.env
                continue
            # 无法继续解包
            break
        # 原始环境已解包

        # 3. 保存关键参数
        self.eval_path = eval_path
        self.model_file = model_file
        self.eval_ep_num = eval_ep_num
        self.eval_env = self.cfg.get('options', 'env_name')
        self.eval_dynamics = self.cfg.get('options', 'dynamic_name')
        
        # 4. 目标点序列（用于连续测试）
        # goal_sequence 是一个列表，每个元素是一个 [x, y, z] 目标点
        self.goal_sequence = goal_sequence if goal_sequence is not None else None
        self.use_continuous_goals = (self.goal_sequence is not None and len(self.goal_sequence) > 0)

    def terminate(self):
        """可选：用于外部中断评估（当前未被调用）"""
        print('Evaluation terminated')

    def run(self):
        """
        QThread 入口函数。默认运行 DRL 模型评估。
        注：若需运行规则策略，可取消注释 self.run_rule_policy()。
        """
        # self.run_rule_policy()
        return self.run_drl_model()

    def run_drl_model(self):
        """
        加载 DRL 模型，在环境中运行指定轮次的评估，
        记录轨迹、动作、状态、奖励等，并计算成功率、平均奖励等指标。
        """
        print('Start DRL model evaluation...')

        # 1. 根据配置加载对应算法的模型
        algo = self.cfg.get('options', 'algo')
        if algo == 'TD3':
            model = TD3.load(self.model_file, env=self.env)
        elif algo == 'SAC':
            model = SAC.load(self.model_file, env=self.env)
        elif algo == 'PPO':
            model = PPO.load(self.model_file, env=self.env)
        else:
            raise Exception(f'Unsupported algorithm: {algo}')
        
        self.env.model = model  # （可选）将模型绑定到环境，便于内部使用

        # 2. 初始化评估变量
        obs = self.env.reset()
        
        # 如果是连续目标点模式，设置第一个目标点
        if self.use_continuous_goals:
            first_goal = self.goal_sequence[0]
            print(f'连续目标点模式：设置第一个目标点: {first_goal}')
            obs = self.env.set_new_goal(first_goal)
            # 初始化起点位置（用于计算总路程）
            start_position = self.env.dynamic_model.get_position()
        
        episode_num = 0
        reward_sum = np.array([0.0])          # 每轮累计奖励
        episode_successes = []                # 成功标志列表
        episode_crashes = []                  # 碰撞标志列表
        step_num_list = []                    # 成功时的步数（用于计算平均完成时间）
        episode_distances = []                # 每轮的总路程（连续目标点模式）

        # 存储所有 episode 的数据（用于后续分析/绘图）
        traj_list_all = []
        action_list_all = []
        state_list_all = []
        obs_list_all = []

        # 当前 episode 的临时缓存
        traj_list = []
        action_list = []
        state_raw_list = []
        obs_list = []

        # 连续目标点测试相关变量
        goal_index = 0  # 当前目标点索引（从0开始，对应第一个目标点）
        goal_success_count = 0  # 成功到达的目标点数量
        total_distance = 0.0  # 总路程（累加实际飞行距离）
        last_position = None  # 上一个位置（用于计算距离）
        
        # 如果是连续目标点模式，初始化起点位置
        if self.use_continuous_goals:
            last_position = self.env.dynamic_model.get_position().copy()

        # 注意：cv2.waitKey() 在无 imshow 时可能阻塞或无效，建议删除
        # cv2.waitKey()  # ← 可安全移除

        # 3. 开始评估循环
        while episode_num < self.eval_ep_num:
            # 使用确定性策略进行推理
            unscaled_action, _ = model.predict(obs, deterministic=True)

            # 执行一步
            new_obs, reward, done, info = self.env.step(unscaled_action)

            # 记录当前状态
            pose = self.env.dynamic_model.get_position()  # 获取无人机位置
            traj_list.append(pose)
            action_list.append(unscaled_action)
            state_raw_list.append(self.env.dynamic_model.state_raw)  # 原始状态（如速度、角度等）
            obs_list.append(obs)
            
            # 计算并累加路程（连续目标点模式）
            if self.use_continuous_goals:
                if last_position is not None:
                    # 计算两点之间的欧氏距离（2D距离，忽略z轴）
                    distance = np.sqrt((pose[0] - last_position[0])**2 + (pose[1] - last_position[1])**2)
                    total_distance += distance
                last_position = pose.copy()

            obs = new_obs
            reward_sum[-1] += reward

            # Episode 结束处理
            if done:
                is_success = info.get('is_success', False)
                is_crash = info.get('is_crash', False)
                is_not_in_workspace = info.get('is_not_in_workspace', False)
                step_num_info = info.get('step_num', 0)

                # 如果是连续目标点模式且成功到达目标点
                if self.use_continuous_goals and is_success and goal_index < len(self.goal_sequence) - 1:
                    # 移动到下一个目标点，不重置环境
                    goal_index += 1
                    goal_success_count += 1
                    next_goal = self.goal_sequence[goal_index]
                    print(f'[连续模式] 到达目标点 {goal_index}/{len(self.goal_sequence)}, 切换到: {next_goal}')
                    
                    # 设置新目标点
                    obs = self.env.set_new_goal(next_goal)
                    
                    # 在连续目标点模式下，增加最大步数限制，给所有目标点足够的步数
                    # 保存原始最大步数（只在第一次保存）
                    if not hasattr(self.env, '_original_max_episode_steps'):
                        self.env._original_max_episode_steps = self.env.max_episode_steps
                    
                    # 不重置 step_num，让它继续累计
                    # 但增加 max_episode_steps，为每个目标点分配足够的步数
                    remaining_goals = len(self.goal_sequence) - goal_index
                    steps_per_goal = self.env._original_max_episode_steps
                    
                    # 计算新的最大步数
                    new_max_steps = self.env.step_num + 1 + (remaining_goals + 1) * steps_per_goal * 3
                    # 使用自定义属性名存储更新后的值，避免被覆盖
                    self.env.max_episode_steps = new_max_steps
                    self.env.__dict__['max_episode_steps'] = new_max_steps
                    # 在原始环境实例上设置
                    self.raw_env._actual_max_episode_steps = new_max_steps
                    self.env._actual_max_episode_steps = new_max_steps
                    
                    # 继续循环，不结束当前 episode（下次循环时 done 会重新计算，应该为 False）
                    continue
                else:
                    # 正常结束 episode（非连续模式，或连续模式下到达最后一个目标点，或发生碰撞/超时）
                    episode_num += 1
                    if is_success:
                        goal_success_count += 1
                    
                    # 打印详细的结束原因
                    end_reason = "成功到达最后一个目标点" if (is_success and self.use_continuous_goals) else \
                                 ("成功" if is_success else "碰撞" if is_crash else \
                                 ("超出工作空间" if is_not_in_workspace else "超时"))
                    
                    # 如果是连续目标点模式且成功完成所有目标点，输出总路程
                    if self.use_continuous_goals and is_success and goal_success_count == len(self.goal_sequence):
                        print(f'Episode {episode_num} 结束: reward={reward_sum[-1]:.2f}, success={is_success}, '
                              f'goals_reached={goal_success_count}/{len(self.goal_sequence)}, '
                              f'结束原因: {end_reason}, step_num: {step_num_info}')
                        print(f'总路程: {total_distance:.2f} 米')
                        # 记录成功完成的总路程
                        episode_distances.append(total_distance)
                    else:
                        print(f'Episode {episode_num} 结束: reward={reward_sum[-1]:.2f}, success={is_success}, '
                              f'goals_reached={goal_success_count if self.use_continuous_goals else (1 if is_success else 0)}, '
                              f'结束原因: {end_reason}, step_num: {step_num_info}')
                        # 即使失败也记录路程（如果连续模式）
                        if self.use_continuous_goals:
                            episode_distances.append(total_distance)

                    episode_successes.append(float(is_success))
                    episode_crashes.append(float(is_crash))

                    # 新开一轮奖励记录
                    reward_sum = np.append(reward_sum, 0.0)

                    # 用特殊标记区分 episode 结束原因（1=成功, 2=碰撞, 3=超时等）
                    if is_success:
                        traj_list.append(1)
                        action_list.append(1)
                        step_num_list.append(info.get('step_num', 0))
                    elif is_crash:
                        traj_list.append(2)
                        action_list.append(2)
                    else:
                        traj_list.append(3)
                        action_list.append(3)

                    # 保存本轮数据
                    traj_list_all.append(traj_list)
                    action_list_all.append(action_list)
                    state_list_all.append(state_raw_list)
                    obs_list_all.append(obs_list)

                    # 重置临时缓存
                    traj_list = []
                    action_list = []
                    state_raw_list = []
                    obs_list = []

                    # 重置连续目标点相关变量
                    goal_index = 0
                    goal_success_count = 0
                    total_distance = 0.0
                    last_position = None
                    
                    # 恢复原始的 max_episode_steps（如果被修改过）
                    if hasattr(self.env, '_original_max_episode_steps'):
                        self.env.max_episode_steps = self.env._original_max_episode_steps
                        delattr(self.env, '_original_max_episode_steps')
                    # 清除 _actual_max_episode_steps（在原始环境和包装器上）
                    if hasattr(self.raw_env, '_actual_max_episode_steps'):
                        delattr(self.raw_env, '_actual_max_episode_steps')
                    if hasattr(self.env, '_actual_max_episode_steps'):
                        delattr(self.env, '_actual_max_episode_steps')

                    # 重置环境
                    obs = self.env.reset()
                    
                    # 如果是连续目标点模式，重置后设置第一个目标点
                    if self.use_continuous_goals:
                        first_goal = self.goal_sequence[0]
                        obs = self.env.set_new_goal(first_goal)
                        # 重新初始化起点位置
                        last_position = self.env.dynamic_model.get_position().copy()

        # 4. 保存评估结果到磁盘
        eval_folder = f"{self.eval_path}/eval_{self.eval_ep_num}_{self.eval_env}_{self.eval_dynamics}"
        os.makedirs(eval_folder, exist_ok=True)

        np.save(f"{eval_folder}/traj_eval", np.array(traj_list_all, dtype=object))
        np.save(f"{eval_folder}/action_eval", np.array(action_list_all, dtype=object))
        np.save(f"{eval_folder}/state_eval", np.array(state_list_all, dtype=object))
        np.save(f"{eval_folder}/obs_eval", np.array(obs_list_all, dtype=object))

        # 5. 打印并保存汇总指标
        avg_reward = reward_sum[:self.eval_ep_num].mean()
        success_rate = np.mean(episode_successes)
        crash_rate = np.mean(episode_crashes)
        avg_step_success = np.mean(step_num_list) if step_num_list else float('nan')
        
        # 如果是连续目标点模式，计算平均路程
        if self.use_continuous_goals and len(episode_distances) > 0:
            avg_distance = np.mean(episode_distances)
            min_distance = np.min(episode_distances)
            max_distance = np.max(episode_distances)
            std_distance = np.std(episode_distances)
            
            print(f'\n{"="*60}')
            print(f'评估结果汇总 (共 {self.eval_ep_num} 个 episodes):')
            print(f'{"="*60}')
            print(f'平均奖励: {avg_reward:.2f}')
            print(f'成功率: {success_rate:.2%} ({np.sum(episode_successes)}/{self.eval_ep_num})')
            print(f'碰撞率: {crash_rate:.2%} ({np.sum(episode_crashes)}/{self.eval_ep_num})')
            print(f'平均步数(成功): {avg_step_success:.1f}')
            print(f'\n连续目标点模式统计:')
            print(f'  平均总路程: {avg_distance:.2f} 米')
            print(f'  最短总路程: {min_distance:.2f} 米')
            print(f'  最长总路程: {max_distance:.2f} 米')
            print(f'  路程标准差: {std_distance:.2f} 米')
            print(f'  目标点数量: {len(self.goal_sequence)}')
            print(f'{"="*60}')
            
            results = [avg_reward, success_rate, crash_rate, avg_step_success, 
                      avg_distance, min_distance, max_distance, std_distance]
        else:
            print(f'\n{"="*60}')
            print(f'评估结果汇总 (共 {self.eval_ep_num} 个 episodes):')
            print(f'{"="*60}')
            print(f'平均奖励: {avg_reward:.2f}')
            print(f'成功率: {success_rate:.2%} ({np.sum(episode_successes)}/{self.eval_ep_num})')
            print(f'碰撞率: {crash_rate:.2%} ({np.sum(episode_crashes)}/{self.eval_ep_num})')
            print(f'平均步数(成功): {avg_step_success:.1f}')
            print(f'{"="*60}')
            
            results = [avg_reward, success_rate, crash_rate, avg_step_success]

        print("\nFinal results:", results)
        np.save(f"{eval_folder}/results", np.array(results))

        return results

    def run_rule_policy(self):
        """
        使用规则策略（而非 DRL 模型）进行评估，用于消融实验或对比。
        """
        obs = self.env.reset()
        episode_num = 0
        reward_sum = np.array([0.0])

        while episode_num < self.eval_ep_num:
            unscaled_action = rule_based_policy(obs)
            new_obs, reward, done, info = self.env.step(unscaled_action)
            reward_sum[-1] += reward
            obs = new_obs

            if done:
                episode_num += 1
                is_success = info.get('is_success', False)
                print(f'Episode {episode_num}: reward={reward_sum[-1]:.2f}, success={is_success}')
                reward_sum = np.append(reward_sum, 0.0)
                obs = self.env.reset()


# ────────────────────────────────────────────────
# 单模型评估入口（命令行调试用）
# ────────────────────────────────────────────────
def main():
    """用于快速测试单个模型（无 GUI）"""
    eval_path = r'C:\Users\helei\Documents\GitHub\UAV_Navigation_DRL_AirSim\logs_new\Trees\2022_12_02_21_46_SimpleMultirotor_mlp_SAC'
    config_file = eval_path + '/config/config.ini'
    model_file = eval_path + '/models/model_sb3.zip'
    eval_ep_num = 50

    evaluate_thread = EvaluateThread(eval_path, config_file, model_file, eval_ep_num)
    evaluate_thread.run()  # 直接同步运行（非线程模式）


# ────────────────────────────────────────────────
# 批量评估多个模型（常用于论文实验）
# ────────────────────────────────────────────────
def run_eval_multi():
    """
    遍历 logs_eval/ 下的所有模型目录，批量运行评估。
    适用于消融实验、不同环境/动力学下的泛化性测试。
    """
    eval_logs_name = 'Maze'                     # 数据集名称（对应 logs_eval/Maze/）
    eval_logs_path = f'logs_eval/{eval_logs_name}'
    eval_ep_num = 50
    eval_env_name = 'NH_center'                 # 评估环境（可覆盖训练时环境）
    eval_dynamic_name = 'SimpleMultirotor'      # 评估用动力学模型

    # 收集所有模型路径（假设结构为 logs_eval/name/train_id/repeat_id/）
    model_list = []
    for train_name in os.listdir(eval_logs_path):
        train_dir = os.path.join(eval_logs_path, train_name)
        if not os.path.isdir(train_dir):
            continue
        for repeat_name in os.listdir(train_dir):
            model_path = os.path.join(train_dir, repeat_name)
            if os.path.isdir(model_path):
                model_list.append(model_path)

    # 逐个评估
    results_list = []
    for i in tqdm(range(len(model_list)), desc="Evaluating models"):
        eval_path = model_list[i]
        config_file = os.path.join(eval_path, 'config/config.ini')
        model_file = os.path.join(eval_path, 'models/model_sb3.zip')

        print(f"[{i}] Evaluating: {eval_path}")
        try:
            evaluate_thread = EvaluateThread(
                eval_path, config_file, model_file,
                eval_ep_num, eval_env_name, eval_dynamic_name
            )
            results = evaluate_thread.run()
            results_list.append(results)
        except Exception as e:
            print(f"❌ Failed: {e}")
            results_list.append([np.nan, np.nan, np.nan, np.nan])

    # 保存所有结果
    save_path = f'logs_eval/results/eval_{eval_ep_num}_{eval_logs_name}_{eval_env_name}_{eval_dynamic_name}.npy'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, np.array(results_list))
    print(f"✅ All results saved to: {save_path}")


# ────────────────────────────────────────────────
# 程序入口
# ────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        # main()           # ← 单模型测试
        run_eval_multi()   # ← 批量评估（当前启用）
    except KeyboardInterrupt:
        print('User interrupted. Exiting gracefully.')