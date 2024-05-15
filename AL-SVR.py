#相关库导入
import numpy as np
import pandas as pd
import h5py
import heapq
import matplotlib.pyplot as plt
from scipy.stats import norm
import gmpy2
from sklearn import preprocessing #预处理归一化
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp
from sklearn.model_selection import cross_val_score
import time

import warnings
import sklearn.utils.validation as validation
warnings.filterwarnings("ignore", category=validation.DataConversionWarning)

#数据集导入 176组数据
data = pd.read_excel(r"D:\课程资料\本科毕设\本科毕业设计\简化硬度数据集.xlsx")
X = data.iloc[:,0:6].values
y = data.iloc[:,6].values
    #数据要先归一化
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1)) #最值归一化
X_minmax = min_max_scaler.fit_transform(X)
def EI(regressor, X_train, y_train, X_space, y_space):
    n_samples = int(2 * X.shape[0])  # 抽取的样本数量
    y_max = max(y_train)
    stds = []
    y_preds = []
    EIs = []
    for x in X_space:
        L = []
        for i in range(50):
            X_train_bootstrap, y_train_bootstrap = resample(X_train, y_train, n_samples=n_samples, random_state=i)
            regressor.fit(X_train_bootstrap, y_train_bootstrap)
            y_pred_bootstrap = regressor.predict(x.reshape(1, -1))  # 第一个参数是行数，第二个参数是列数
            L.append(y_pred_bootstrap)
        std = np.std(L, axis=0)
        stds.append(std)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(x.reshape(1, -1))
        y_preds.append(y_pred)
        z = (y_pred - y_max) / std
        EI = std * (norm.pdf(z) + z * norm.cdf(z)) # pdf-概率密度函数；cdf-累计分布函数
        EIs.append(EI)
    query_idx = np.argmax(EIs)
    return X_space[query_idx], y_space[query_idx], query_idx, y_preds[query_idx], stds[query_idx], EIs[query_idx]
def value_weighting(regressor, X_train, y_train, X_space, y_space):
    n_samples = int(2 * X.shape[0])  # 抽取的样本数量
    values = []
    stds = []
    y_preds = []
    for x in X_space:
        L = []
        for i in range(50):
            X_train_bootstrap, y_train_bootstrap = resample(X_train, y_train, n_samples=n_samples, random_state=i)
            regressor.fit(X_train_bootstrap, y_train_bootstrap)
            y_pred_bootstrap = regressor.predict(x.reshape(1, -1))  # 第一个参数是行数，第二个参数是列数
            L.append(y_pred_bootstrap)
        std = np.std(L, axis=0)
        stds.append(std)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(x.reshape(1, -1))
        y_preds.append(y_pred)
        value = 0.1 * std + 0.9 * y_pred
        values.append(value)
    # print(stds, y_preds)
    query_idx = np.argmax(values)
    return X_space[query_idx], y_space[query_idx], query_idx, y_preds[query_idx], stds[query_idx], values[query_idx]
def rank_weighting(regressor, X_train, y_train, X_space, y_space):
    n_samples = int(2 * X.shape[0])  # 抽取的样本数量
    rank_values=[]
    stds= []
    y_preds = []
    for x in X_space:
        L = []
        for i in range(50):
            X_train_bootstrap, y_train_bootstrap = resample(X_train, y_train, n_samples=n_samples, random_state=i)
            regressor.fit(X_train_bootstrap, y_train_bootstrap)
            y_pred_bootstrap = regressor.predict(x.reshape(1, -1))  # 第一个参数是行数，第二个参数是列数
            L.append(y_pred_bootstrap)
        std = np.std(L, axis=0)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(x.reshape(1, -1))
        stds.append(std)
        y_preds.append(y_pred)
    sorted_stds = sorted(stds, reverse=True)
    sorted_y_preds = sorted(y_preds, reverse=True)
    rank_stds = [None]*len(stds)
    rank_y_preds = [None]*len(y_preds)
    for index, value in enumerate(sorted_stds):
        rank_stds[stds.index(value)] = index + 1
    for index, value in enumerate(sorted_y_preds):
        rank_y_preds[y_preds.index(value)] = index + 1
    for i in range(len(rank_stds)):
        rank_values.append(rank_stds[i]*0.8+rank_y_preds[i]*0.2)
    query_idx = np.argmin(rank_values)
    return X_space[query_idx], y_space[query_idx], query_idx, y_preds[query_idx], stds[query_idx], rank_values[query_idx]

def random_RMSE(model, X, y):   #test_size=0.3
    RMSEs = []
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
        RMSEs.append(RMSE)
    mean_RMSE = np.mean(RMSEs)
    return mean_RMSE
# 定义网格搜索的参数范围
param_grid = {
    'C': np.linspace(50, 100, num=50),
    'gamma': np.linspace(1, 5, num=10)
}


num = 0  #执行一百次
i = 0  #随机种子
interaions = []
while num < 20:
    X_data, X_space, y_data, y_space = train_test_split(X_minmax, y, test_size = 0.8, random_state = i)
    X_data_temp = X_data.reshape(-1,6)
    y_data_temp = y_data.reshape(-1,1)
    X_space_temp = X_space.reshape(-1,6)
    y_space_temp = y_space.reshape(-1,1)
    constant1 = max(y_data_temp)
    constant2 = max(y_space_temp)
    if constant1 <= constant2:
        num += 1
        interation = 0   #迭代次数
        # 使用GridSearchCV进行网格搜索
        grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_data_temp, y_data_temp)
        # 使用最佳参数创建SVR模型
        regressor = grid_search.best_estimator_
        # ####################################################################################################
        while True:
            a, b, c, d, e, f = value_weighting(regressor, X_data_temp, y_data_temp, X_space_temp, y_space_temp)
            interation += 1

            if b > constant1:
                X_data_temp = np.append(X_data_temp, a).reshape(-1,6)
                y_data_temp = np.append(y_data_temp, b).reshape(-1,1)
                X_space_temp = np.delete(X_space_temp, c, axis=0).reshape(-1,6)
                y_space_temp = np.delete(y_space_temp, c).reshape(-1,1)
                print(f"随机种子为{i},迭代{interation}次,已执行{num}次")
                interaions.append(interation)
                break
            else:
                X_data_temp = np.append(X_data_temp, a).reshape(-1,6)
                y_data_temp = np.append(y_data_temp, b).reshape(-1,1)
                X_space_temp = np.delete(X_space_temp, c, axis=0).reshape(-1,6)
                y_space_temp = np.delete(y_space_temp, c).reshape(-1,1)
        i += 1
    else:
        i += 1
df = pd.DataFrame(interaions)
# 指定输出的 Excel 文件路径
output_file_path = r'D:\课程资料\本科毕设\本科毕业设计\师兄小论文数据\value0.9.xlsx'
# 将 DataFrame 写入 Excel 文件
df.to_excel(output_file_path, index=False, sheet_name='Interactions')