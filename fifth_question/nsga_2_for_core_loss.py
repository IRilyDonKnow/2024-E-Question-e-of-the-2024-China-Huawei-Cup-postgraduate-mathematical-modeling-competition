"""
Main codes refer to:
    https://blog.csdn.net/no___good/article/details/134749475?type=blog&rId=134749475&refer=APP&source=mr_unknown23333
    @no__good

Our contributions: 
    1. On his basis, we extended the objective function to a nine-dimensional function.
    2. Plotted the log of Pareto Front.
    3. And plotted the HV diagram of the entire optimization process.
    
Thanks to the open source community:
    github.com
    gitee.com

And chinese website:
    csdn.com
    baidu.com
    
All dependencies on open source python packages are available in the file:
    requirement.txt

And the use of the code is available in the file:
    readme.md
    
"""
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import xgboost as xgb
from pymoo.indicators.hv import HV
 
class Individual(object):
    def __init__(self):
        self.solution = None
        self.objective = defaultdict()
 
        self.n = 0
        self.rank = 0
        self.S = []
        self.distance = 0
 
    def bound_process(self, bound_min, bound_max):
        for i, item in enumerate(self.solution):
            if item > bound_max:
                self.solution[i] = bound_max
            elif item < bound_min:
                self.solution[i] = bound_min
 
    def calculate_objective(self, objective_fun):
        self.objective = objective_fun(self.solution)
 
    def __lt__(self, other):
        v1 = list(self.objective.values())
        v2 = list(other.objective.values())
        for i in range(len(v1)):
            if v1[i] > v2[i]:
                return 0 
        return 1
 
def Generate_individual():
    x = [0]*9
    X_BOUND_0 = [-0.05, 0.07]  # time_skew取值范围
    X_BOUND_1 = [-2.0, -1.0]  # time_kurt取值范围
    X_BOUND_2 = [1.1, 1.7]  # time_form取值范围
    X_BOUND_3 = [0, 0.3]  # max_al取值范围
    X_BOUND_4 = [1.2, 2.2]  # time_impulse取值范围
    X_BOUND_5 = [1.5, 2.5]  # time_clearance取值范围
    X_BOUND_6 = [25, 50,70,90]  # temp取值范围
    X_BOUND_7 = [1,2,3,4]  # materials取值范围
    X_BOUND_8 = [0, 50000]  # freq取值范围
    x[0] = np.random.rand(1)[0] * (X_BOUND_0[1] - X_BOUND_0[0]) + X_BOUND_0[0]
    x[1] = np.random.rand(1)[0] * (X_BOUND_1[1] - X_BOUND_1[0]) + X_BOUND_1[0]
    x[2] = np.random.rand(1)[0] * (X_BOUND_2[1] - X_BOUND_2[0]) + X_BOUND_2[0]
    x[3] = np.random.rand(1)[0] * (X_BOUND_3[1] - X_BOUND_3[0]) + X_BOUND_3[0]
    x[4] = np.random.rand(1)[0] * (X_BOUND_4[1] - X_BOUND_4[0]) + X_BOUND_4[0]
    x[5] = np.random.rand(1)[0] * (X_BOUND_5[1] - X_BOUND_5[0]) + X_BOUND_5[0]
    x[6] = random.choice(X_BOUND_6)
    x[7] = random.choice(X_BOUND_7)
    x[8] = random.randint(X_BOUND_8[0], X_BOUND_8[1])
    #print("x_8_ini: {}".format(x[8]))
    return np.array(x)
 
def main():
    generations = 500# 迭代次数
    popnum = 100  # 种群大小
    eta = 1  # 变异分布参数，该值越大则产生的后代个体逼近父代的概率越大。Deb建议设为 1
 
    # 问题定义
    # poplength = 30  # 单个个体解向量的维数
    # bound_min = 0  # 定义域
    # bound_max = 1
    # objective_fun = ZDT1
 
    poplength = 9  # 单个个体解向量的维数
    bound_min = -5  # 定义域
    bound_max = 5
    objective_fun = KUR  #  目标函数
 
    # 生成第一代种群
    P = [] # 创建一个空列表 P，用于存储个体对象
    for i in range(popnum):
        # 通过循环生成 popnum 个个体，并将它们添加到列表 P 中。
        P.append(Individual())
        P[i].solution = Generate_individual()  # 随机生成个体可行解
        P[i].calculate_objective(objective_fun)  # 计算目标函数值
    fast_non_dominated_sort(P)
    
    Q = make_new_pop(P, eta, bound_min, bound_max, objective_fun)
 
    P_t = P
    Q_t = Q
 
    for gen_cur in range(generations):
        R_t = P_t + Q_t
        F = fast_non_dominated_sort(R_t)
 
        P_n = []
        i = 1
        while len(P_n) + len(F[i]) < popnum:  
            crowding_distance_assignment(F[i])
            P_n = P_n + F[i] 
            i = i + 1
        F[i].sort(key=lambda x: x.distance) 
        P_n = P_n + F[i][:popnum - len(P_n)] 
        Q_n = make_new_pop(P_n, eta, bound_min, bound_max,
                           objective_fun) 
 

        P_t = P_n
        Q_t = Q_n
        plot_P(P_t,gen_cur + 1)
        plt.pause(0.1)
    plt.show()
 
    return 0
 
def fast_non_dominated_sort(P):
    F = defaultdict(list)
 
    for p in P:
        p.S = [] 
        p.n = 0 
        for q in P:
            if p < q:
                p.S.append(q)
            elif q < p:
                p.n += 1 
        if p.n == 0:
            p.rank = 1
            F[1].append(p)
    i = 1
    while F[i]:
        Q = []
        for p in F[i]:
            for q in p.S:
                q.n = q.n - 1
                if q.n == 0:
                    q.rank = i + 1
                    Q.append(q)
        i = i + 1
        F[i] = Q
 
    return F
 

def make_new_pop(P, eta, bound_min, bound_max, objective_fun):
    popnum = len(P)
    Q = []
    for i in range(int(popnum / 2)): 
        i = random.randint(0, popnum - 1)
        j = random.randint(0, popnum - 1)
        parent1 = binary_tournament(P[i], P[j])
 
        i = random.randint(0, popnum - 1)
        j = random.randint(0, popnum - 1)
        parent2 = binary_tournament(P[i], P[j])
 
        while (parent1.solution == parent2.solution).all():
            i = random.randint(0, popnum - 1)
            j = random.randint(0, popnum - 1)
            parent2 = binary_tournament(P[i], P[j])
 
        
        Two_offspring = crossover_mutation(parent1, parent2, eta, bound_min, bound_max, objective_fun)
 
        
        Q.append(Two_offspring[0])
        Q.append(Two_offspring[1])
    return Q
 

def crowding_distance_assignment(L):
    l = len(L) 
 
    for i in range(l):
        L[i].distance = 0
 
    for m in L[0].objective.keys():
        L.sort(key=lambda x: x.objective[m])
        L[0].distance = float('inf')
        L[l - 1].distance = float('inf')
 
        f_max = L[l - 1].objective[m]
        f_min = L[0].objective[m]
 
        try:
            for i in range(1, l - 1):
                L[i].distance = L[i].distance + (L[i + 1].objective[m] - L[i - 1].objective[m]) / (f_max - f_min)
        except Exception:
            print(str(m) + "Max Value: " + str(f_max) + "Min Value: " + str(f_min))
 
def binary_tournament(ind1, ind2):
    if ind1.rank != ind2.rank:
        return ind1 if ind1.rank < ind2.rank else ind2
    elif ind1.distance != ind2.distance:
        return ind1 if ind1.distance > ind2.distance else ind2
    else:
        return ind1
 

def crossover_mutation(parent1, parent2, eta, bound_min, bound_max, objective_fun):
    poplength = len(parent1.solution)
 
    offspring1 = Individual()
    offspring2 = Individual()
    offspring1.solution = np.empty(poplength)
    offspring2.solution = np.empty(poplength)
 
   
    for i in range(poplength):
        rand = random.random()
        beta = (rand * 2) ** (1 / (eta + 1)) if rand < 0.5 else (1 / (2 * (1 - rand))) ** (1.0 / (eta + 1))
        offspring1.solution[i] = 0.5 * ((1 + beta) * parent1.solution[i] + (1 - beta) * parent2.solution[i])
        offspring2.solution[i] = 0.5 * ((1 - beta) * parent1.solution[i] + (1 + beta) * parent2.solution[i])
    for i in range(poplength):
        mu = random.random()
        delta = 2 * mu ** (1 / (eta + 1)) if mu < 0.5 else (1 - (2 * (1 - mu)) ** (1 / (eta + 1)))
        offspring1.solution[i] = offspring1.solution[i] + delta
 
    offspring1.calculate_objective(objective_fun)
    offspring2.calculate_objective(objective_fun)
 
    return [offspring1, offspring2]
 
 
def goal_func_1(test_x):
    x = []
    x.append(test_x)
    x.append(test_x)
    x = np.array(x)
    clf = xgb.XGBRegressor()
    booster = xgb.Booster()
    booster.load_model('best_models/model_xgboost.bin')
    clf._Booster = booster
    model_xgb = clf
    min_value = 2.6186892424572115
    delta_value = 3.9395550971001576
    
    y_pred_x = model_xgb.predict(x)[0]
    y_real_x_1 = np.power(10,y_pred_x *delta_value + min_value)
    
    return np.log(y_real_x_1)

def goal_func_2(test_x):
    y_real_x_2 = test_x[3]*test_x[8]
    return np.log(y_real_x_2)

def pair_floats(float1, float2):
    return [float1,float2]

def calculate_hv(X,Y):
    ref_point = np.array([-13.8, 14])
    ind = HV(ref_point=ref_point)
    A = []
    for i in range(len(X)):
        A_sin = []
        A_sin.append(X[i])
        A_sin.append(Y[i])
        A.append(A_sin)
    A = np.array(A)
    hv = ind(A)
    return hv

def KUR(x):
    f = defaultdict(float) 
 
    f[1] = 0
    f[2] = 0
    
    f[1] = -goal_func_1(x)
    f[2] = goal_func_2(x)
    
    return f
 
def plot_P(P,gen_cur):
    X = []
    Y = []
    for ind in P:
        X.append(ind.objective[1])
        Y.append(ind.objective[2])
    
    plt.figure(dpi = 600)
    fig, ax = plt.subplots(figsize=(8,5),dpi = 600)
    ax.scatter(X, Y, s = 80, marker='o',edgecolors='darkgoldenrod',facecolors='gold', label='NSGA-2:Multi-Objective Optimization')
    plt.xlabel('F1')
    plt.ylabel('F2')
    plt.grid()
    plt.title("Pareto Front Plot Generation:{}".format(gen_cur))
    plt.show()
    hv = calculate_hv(X,Y)
    hv_list.append(hv)

def painting_hv(hv_list):
    x = np.linspace(0,len(hv_list),len(hv_list))
    plt.figure(dpi = 600)
    plt.plot(x, hv_list, color='darkgoldenrod') 
    plt.grid()
    plt.title('Hypervolume with NSGA-II')
    plt.show()
    return 0

# Entering main function
if __name__ == '__main__':
    hv_list = []
    main()
    painting_hv(hv_list)