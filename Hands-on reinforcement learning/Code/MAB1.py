# 导入需要使用的库，其中 numpy 是支持数组和矩阵运算的科学计算库，而 matplotlib 是绘图库
import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    """伯努利多臂老虎机，输入 K 表示拉杆个数"""
    def __init__(self, K):
        self.probs = np.random.uniform(size=K)  # 随机生成 K 个 0~1 的数，作为拉动每根拉杆的获奖概率
        self.best_idx = np.argmax(self.probs)  # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx]  # 最大的获奖概率
        self.K = K

    def step(self, k):
        # 当玩家选择了 k 号拉杆后，根据拉动该老虎机的 k 号拉杆获得奖励的概率返回 1（获奖）或 0（未获奖）
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0

np.random.seed(1)  # 设定随机种子，使实验具有可重复性

K = 10
bandit_10_arm = BernoulliBandit(K)
print("随机生成了一个%d臂伯努利老虎机" % K)
print("获奖概率最大的拉杆为%d号，其获奖概率为%.4f" % (bandit_10_arm.best_idx, bandit_10_arm.best_prob))