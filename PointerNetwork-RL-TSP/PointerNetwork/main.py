#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import DataGenerator
from actor import Actor
from config import get_config, print_config

def main():
    # Get running configuration
    config, _ = get_config()
    print_config()

    # Build tensorflow graph from config
    print("Building graph...")
    actor = Actor(config)

    # Saver to save & restore all the variables.
    variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
    saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)

    rewards = []

    result_pos_list = []

    print("Starting session...")
    with tf.Session() as sess:
        # Run initialize op
        sess.run(tf.global_variables_initializer())

        # Restore variables from disk.
        if config.restore_model is True:
            saver.restore(sess, config.restore_from)
            print("Model restored.")

        training_set = DataGenerator(config)

        # training mode
        if config.training_mode:

            print("Starting training...")
            for i in tqdm(range(config.iteration)):
                # Get feed dict
                input_batch = training_set.train_batch()
                feed = {actor.input_: input_batch}

                # Forward pass & train step
                positions, reward, train_step1, train_step2 = sess.run(
                    [actor.positions, actor.reward, actor.train_step1, actor.train_step2],
                    feed_dict=feed)

                rewards.append(np.mean(reward))

                if i % 100 == 0 and i != 0:
                    print("after " + str(i) + " rounds training, Travel Distance is: " + str(rewards[-1]))

                # Save the variables to disk
                if i % 1000 == 0 and i != 0:
                    save_path = saver.save(sess, config.save_to)
                    print("Model saved in file: %s" % save_path)

            print("Training COMPLETED !")
            save_path = saver.save(sess, config.save_to)
            print("Model saved in file: %s" % save_path)

        # test mode
        else:
            # Get test data
            input_batch = training_set.test_batch()
            feed = {actor.input_: input_batch}

            # Sample solutions
            positions, reward, _, _ = sess.run(
                [actor.positions, actor.reward, actor.train_step1, actor.train_step2],
                feed_dict=feed)

            city = input_batch[1]
            print("原始城市坐标:")
            print(city)
            position = positions[1]
            result_pos_list = city[position, :]
            
            # 获取路径长度（reward 就是路径总长度）
            # reward 是 [batch_size] 形状的数组，取索引1的样本（与 city 和 position 保持一致）
            path_length = float(reward[1]) if len(reward) > 1 else float(reward[0]) if len(reward) > 0 else 0.0
            print(f"\n{'='*60}")
            print(f"TSP 路径规划结果:")
            print(f"{'='*60}")
            print(f"城市数量: {len(result_pos_list)}")
            print(f"最短路径长度: {path_length:.4f}")
            print(f"{'='*60}\n")
            
            # 保存原始城市坐标和优化后的路径顺序
            # 原始城市坐标 (x, y)
            original_cities = city  # shape: (max_length, 2)
            # 优化后的路径顺序（按照访问顺序排列的城市坐标）
            optimized_path = result_pos_list  # shape: (max_length, 2)
            
            # 保存为 numpy 文件，供无人机评估使用
            # 格式: 每个目标点 [x, y, z]，z 固定为 5（无人机飞行高度）
            goal_sequence = []
            for i in range(len(optimized_path)):
                x, y = optimized_path[i, 0], optimized_path[i, 1]
                goal_sequence.append([float(x), float(y), 5.0])  # 添加 z 坐标
            
            save_path = './save/goal_sequence.npy'
            np.save(save_path, np.array(goal_sequence))
            print(f"Goal sequence saved to {save_path}")
            print(f"Goal sequence shape: {np.array(goal_sequence).shape}")
            print(f"Goal sequence:\n{goal_sequence}")
            print(f"\n{'='*60}")
            print(f"最终输出: 规划的最短路径长度 = {path_length:.4f}")
            print(f"{'='*60}")

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.rcParams['savefig.dpi'] = 400  # 图片像素
    plt.rcParams['figure.dpi'] = 400  # 分辨率

    if config.training_mode:
        fig = plt.figure()
        plt.plot(list(range(len(rewards))), rewards, c='red')
        plt.title(u"效果曲线")
        plt.xlabel('轮数')
        plt.legend()
        fig.show()
        input("Press Enter to close...")
    else:
        fig = plt.figure()
        plt.plot(result_pos_list[0:config.max_length-1, 0], result_pos_list[0:config.max_length-1, 1], 'o-r')
        plt.title(u"路线")
        plt.legend()
        fig.show()
        input("Press Enter to close...")

if __name__ == "__main__":
    main()
