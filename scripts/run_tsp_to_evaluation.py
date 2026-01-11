#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TSP 路径规划 + 无人机评估的完整流程脚本

流程：
1. 运行 TSP main.py 生成优化后的路径坐标
2. 自动加载生成的坐标
3. 运行无人机评估程序进行测试
"""

import sys
import os
import subprocess
import numpy as np
import time
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
tsp_dir = project_root / 'PointerNetwork-RL-TSP' / 'PointerNetwork'
scripts_dir = current_dir

# 配置路径
TSP_MAIN_SCRIPT = tsp_dir / 'main.py'
TSP_GOAL_FILE = tsp_dir / 'save' / 'goal_sequence.npy'
EVAL_SCRIPT = scripts_dir / 'start_evaluate_with_plot.py'


def run_tsp_generation():
    """
    运行 TSP main.py 生成优化后的路径坐标
    使用 subprocess 方式，更可靠
    """
    print("=" * 60)
    print("步骤 1: 运行 TSP 路径规划生成坐标...")
    print("=" * 60)
    
    if not TSP_MAIN_SCRIPT.exists():
        raise FileNotFoundError(f"TSP main.py 不存在: {TSP_MAIN_SCRIPT}")
    
    # 切换到 TSP 目录
    original_cwd = os.getcwd()
    os.chdir(tsp_dir)
    
    try:
        # 运行 TSP main.py（测试模式）
        # 注意：需要确保在正确的环境中运行（可能需要激活 conda 环境）
        cmd = [sys.executable, str(TSP_MAIN_SCRIPT), '--training_mode', 'False']
        print(f"执行命令: {' '.join(cmd)}")
        print(f"工作目录: {os.getcwd()}")
        print(f"Python 解释器: {sys.executable}")
        print()
        
        # 实时输出，不捕获（这样可以看到进度）
        result = subprocess.run(
            cmd,
            timeout=300,  # 5分钟超时
            check=False  # 不自动抛出异常，我们自己处理
        )
        
        if result.returncode != 0:
            print(f"\n❌ TSP 运行失败，返回码: {result.returncode}")
            raise RuntimeError(f"TSP 生成失败")
        
        # 检查文件是否生成
        if not TSP_GOAL_FILE.exists():
            raise FileNotFoundError(f"坐标文件未生成: {TSP_GOAL_FILE}")
        
        # 验证文件内容
        goal_sequence = np.load(TSP_GOAL_FILE)
        print(f"\n✓ TSP 坐标生成成功!")
        print(f"  文件路径: {TSP_GOAL_FILE}")
        print(f"  目标点数量: {len(goal_sequence)}")
        print(f"  坐标范围: x=[{goal_sequence[:, 0].min():.1f}, {goal_sequence[:, 0].max():.1f}], "
              f"y=[{goal_sequence[:, 1].min():.1f}, {goal_sequence[:, 1].max():.1f}]")
        print(f"  前5个目标点:\n{goal_sequence[:5]}")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("\n❌ TSP 运行超时（超过5分钟）")
        return False
    except Exception as e:
        print(f"\n❌ TSP 运行出错: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        os.chdir(original_cwd)


def run_evaluation():
    """
    运行无人机评估程序
    """
    print("\n" + "=" * 60)
    print("步骤 2: 运行无人机评估程序...")
    print("=" * 60)
    
    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(f"评估脚本不存在: {EVAL_SCRIPT}")
    
    if not TSP_GOAL_FILE.exists():
        raise FileNotFoundError(f"坐标文件不存在，请先运行 TSP 生成: {TSP_GOAL_FILE}")
    
    # 切换到脚本目录
    original_cwd = os.getcwd()
    os.chdir(scripts_dir)
    
    try:
        # 运行评估脚本
        print(f"执行评估脚本: {EVAL_SCRIPT}")
        print(f"工作目录: {os.getcwd()}")
        print("\n提示: 评估程序会启动 GUI 界面，请在 GUI 中查看结果...")
        
        # 直接运行，不捕获输出（因为 GUI 需要交互）
        subprocess.run([sys.executable, str(EVAL_SCRIPT)])
        
    finally:
        os.chdir(original_cwd)


def main():
    """
    主函数：串联整个流程
    """
    print("\n" + "=" * 60)
    print("TSP 路径规划 + 无人机评估完整流程")
    print("=" * 60)
    print(f"项目根目录: {project_root}")
    print(f"TSP 目录: {tsp_dir}")
    print(f"脚本目录: {scripts_dir}")
    print()
    
    try:
        # 步骤 1: 运行 TSP 生成坐标
        success = run_tsp_generation()
        
        if not success:
            print("❌ TSP 坐标生成失败，终止流程")
            return
        
        # 等待一下，确保文件写入完成
        time.sleep(1)
        
        # 步骤 2: 运行评估
        print("\n等待 2 秒后开始评估...")
        time.sleep(2)
        run_evaluation()
        
        print("\n" + "=" * 60)
        print("✓ 完整流程执行完成!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n用户中断执行")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

