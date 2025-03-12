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

class Solver:
    """多臂老虎机算法基本框架"""
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # 每根拉杆的尝试次数
        self.regret = 0  # 当前步的累积后悔
        self.actions = []  # 维护一个列表，记录每一步的动作
        self.regrets = []  # 维护一个列表，记录每一步的累积后悔

    def update_regret(self, k):
        # 计算累积后悔并保存，k为本次动作选择的拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆，由每个具体的策略实现
        raise NotImplementedError

    def run(self, num_steps):
        # 运行一定次数，num_steps为总运行次数
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

class ThompsonSampling(Solver):
    """"汤普森采样算法, 继承 Solver 类"""
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)  # 列表, 表示每根拉杆奖励为 1 的次数
        self._b = np.ones(self.bandit.K)  # 列表, 表示每根拉杆奖励为 0 的次数

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)  # 按照 Beta 分布采样一组奖励样本
        k = np.argmax(samples)  # 选出采样奖励最大的拉杆
        r = self.bandit.step(k)
        self._a[k] += r  # 更新 Beta 分布的第一个参数
        self._b[k] += (1 - r)  # 更新 Beta 分布的第二个参数
        return k

def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入 solvers 是一个列表,列表中的每个元素是一种特定的策略。
    而 solver_names 也是一个列表，存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

np.random.seed(1)  # 设定随机种子，使实验具有可重复性
K = 10
bandit_10_arm = BernoulliBandit(K)
np.random.seed(1)
thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)
print('汤普森采样算法的累积懊悔为:', thompson_sampling_solver.regret)
plot_results([thompson_sampling_solver], ["ThompsonSampling"])