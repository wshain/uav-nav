import numpy as np


class DataGenerator(object):

    # Initialize a DataGenerator
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        
        # 定义障碍物信息
        # 8个正方形障碍物：中心坐标和边长
        self.square_obstacles = [
            (10, 10, 10),      # (中心x, 中心y, 边长)
            (-10, 10, 10),
            (-10, -10, 10),
            (10, -10, 10),
            (0, -30, 10),
            (0, 30, 10),
            (30, 0, 10),
            (-30, 0, 10)
        ]
        
        # 4个圆形障碍物：中心坐标和半径
        self.circle_obstacles = [
            (25, 25, 5),       # (中心x, 中心y, 半径)
            (25, -25, 5),
            (-25, 25, 5),
            (-25, -25, 5)
        ]
        
        # 安全距离：生成的点必须离障碍物至少这个距离（避免触碰障碍物边界）
        self.safety_margin = 5.0  # 安全距离，可以根据需要调整
        
        # 地图范围（开区间，不包括边界）
        self.map_bounds = [-60, 60]  # 实际生成时使用 (-60, 60)，不包括边界

    def is_point_in_obstacle(self, x, y):
        """
        检查点 (x, y) 是否在任何障碍物内或太靠近障碍物（考虑安全距离）
        
        Args:
            x: x 坐标
            y: y 坐标
            
        Returns:
            True 如果点在障碍物内或太靠近障碍物，False 否则
        """
        # 检查正方形障碍物（扩大检测范围，包括安全距离）
        for cx, cy, side in self.square_obstacles:
            half_side = side / 2.0
            # 扩大检测范围：边长 + 2 * 安全距离
            expanded_half_side = half_side + self.safety_margin
            if (cx - expanded_half_side <= x <= cx + expanded_half_side and 
                cy - expanded_half_side <= y <= cy + expanded_half_side):
                return True
        
        # 检查圆形障碍物（扩大检测范围，包括安全距离）
        for cx, cy, radius in self.circle_obstacles:
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            # 扩大检测范围：半径 + 安全距离
            expanded_radius = radius + self.safety_margin
            if distance <= expanded_radius:
                return True
        
        return False

    # Generate random batch for training procedure
    def train_batch(self):
        input_batch = []
        for _ in range(self.batch_size):
            # Generate random TSP instance
            input_ = self.gen_instance()
            # Store batch
            input_batch.append(input_)
        return input_batch

    # Generate random batch for testing procedure
    def test_batch(self):
        # Generate random TSP instance
        input_ = self.gen_instance()
        # Store batch
        input_batch = np.tile(input_, (self.batch_size, 1, 1))
        return input_batch

    # Generate random TSP-TW instance
    def gen_instance(self):
        """
        随机生成 (max_length) 个城市坐标，避开障碍物和地图边界
        环境范围: (-60, 60) x (-60, 60)，不包括边界
        生成的点会保持安全距离，远离障碍物和边界
        """
        x_list = []
        y_list = []
        
        # 计算考虑安全距离后的有效范围
        # 地图边界是 (-60, 60)，加上安全距离后，有效范围是 (-60 + margin, 60 - margin)
        safe_min = int(self.map_bounds[0] + self.safety_margin + 1)  # 向上取整，确保是整数
        safe_max = int(self.map_bounds[1] - self.safety_margin)     # 向下取整，确保是整数
        
        max_attempts = 1000  # 每个点的最大尝试次数
        
        for _ in range(self.max_length):
            attempts = 0
            while attempts < max_attempts:
                # 随机生成坐标（在安全范围内，远离边界和障碍物）
                x = np.random.randint(low=safe_min, high=safe_max + 1)
                y = np.random.randint(low=safe_min, high=safe_max + 1)
                
                # 检查是否在障碍物内（障碍物检测已经包含了安全距离）
                if not self.is_point_in_obstacle(x, y):
                    x_list.append(x)
                    y_list.append(y)
                    break
                
                attempts += 1
            
            # 如果尝试次数过多仍未找到有效点，在安全范围内随机选择
            if attempts >= max_attempts:
                # 在安全范围内随机生成点
                x = np.random.randint(low=safe_min, high=safe_max + 1)
                y = np.random.randint(low=safe_min, high=safe_max + 1)
                
                # 如果这个点也在障碍物内，尝试安全范围内的其他点
                if self.is_point_in_obstacle(x, y):
                    # 在安全范围的中心区域生成点（更可能避开障碍物）
                    center_min = int((safe_min + safe_max) / 2 - 10)
                    center_max = int((safe_min + safe_max) / 2 + 10)
                    center_min = max(center_min, safe_min)
                    center_max = min(center_max, safe_max)
                    x = np.random.randint(low=center_min, high=center_max + 1)
                    y = np.random.randint(low=center_min, high=center_max + 1)
                
                x_list.append(x)
                y_list.append(y)
        
        x = np.array(x_list).reshape(-1, 1)
        y = np.array(y_list).reshape(-1, 1)
        sequence = np.concatenate((x, y), axis=1)

        return sequence
