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

from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Product

from sklearn.model_selection import train_test_split
import warnings
import sklearn.utils.validation as validation
warnings.filterwarnings("ignore", category=validation.DataConversionWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

#数据集导入 176组数据
data = pd.read_excel(r"D:\课程资料\本科毕设\本科毕业设计\简化硬度数据集.xlsx")
X = data.iloc[:,0:6].values.reshape(-1,6)
y = data.iloc[:,6].values.reshape(-1,1)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1)) #最值归一化

def EI(regressor, X_train, y_train, X_space, y_space):
    regressor.fit(X_train, y_train)
    y_max = max(y_train)
    stds = []
    y_preds = []
    EIs = []
    for x in X_space:
        y_pred, std = regressor.predict(x.reshape(1, -1), return_std = True)      #GPR可计算不确定度
        y_preds.append(y_pred)
        stds.append(std)
        z = (y_pred - y_max) / std
        EI = std * (norm.pdf(z) + z * norm.cdf(z)) # pdf-概率密度函数；cdf-累计分布函数
        EIs.append(EI)
    # print(y_preds,'\n',stds)
    query_idx = np.argmax(EIs)
    return X_space[query_idx], y_space[query_idx], query_idx, y_preds[query_idx], stds[query_idx], EIs[query_idx]

# X_data, X_space, y_data, y_space = train_test_split(X_minmax, y_minmax, test_size=0.8, random_state=0)
# ##########################################################################################################
# x_noise_std = 0.001
# np.random.seed(0)
# noise = np.random.normal(0, x_noise_std, size=X_data.shape)
# # noise = np.maximum(noise, 0)#截断正态分布
# X_noisy_data = X_data + noise
# X_combined_data = np.vstack((X_data, X_noisy_data))
# X_data_temp = X_combined_data.reshape(-1,6)
#
# y_noise_std = 2
# np.random.seed(0)
# noise = np.random.normal(0, y_noise_std, size=y_data.shape)
# # noise = np.maximum(noise, 0)#截断正态分布
# y_noisy_data = y_data + noise
# y_combined_data = np.vstack((y_data, y_noisy_data))
# y_data_temp = y_combined_data.reshape(-1,1)
# print(X_data_temp,y_data_temp)
#########################################################################################################
# X1 = X_data
# y1 = y_data
# X_data_temp = np.vstack((X_data, X1)).reshape(-1,6)
# y_data_temp = np.vstack((y_data, y1)).reshape(-1,1)
##########################################################################################################
def add_noise(data, noise_std, i):
    np.random.seed(i)  # 注意：在实际应用中，您可能不希望固定随机种子
    noise = np.random.normal(0, noise_std, size=data.shape)
    return data + noise


num = 0  #执行一百次
i = 0  #随机种子
interaions = []
while num < 100:
    X_data, X_space, y_data, y_space = train_test_split(X, y, test_size=0.8, random_state=i)
    np.random.seed(i)
    x_noise_std = 0.001
    noise = np.random.normal(0, x_noise_std, size=X_data.shape)
    X_noisy_data = X_data + noise
    # X_combined_data = np.vstack((X_data, X_noisy_data))
    y_noise_std = 3
    noise = np.random.normal(0, y_noise_std, size=y_data.shape)
    y_noisy_data = y_data + noise
    # y_combined_data = np.vstack((y_data, y_noisy_data))

    # 合并数据
    X_all = np.vstack((X_noisy_data, X_space))
    min_values = np.min(X_all, axis=0)
    max_values = np.max(X_all, axis=0)
    X_normalized_all = (X_all - min_values) / (max_values - min_values)
    n_samples_combined = X_noisy_data.shape[0]
    # 分割归一化后的数据
    X_data_temp = X_normalized_all[:n_samples_combined, :]
    X_space_temp = X_normalized_all[n_samples_combined:, :]

    y_all = np.vstack((y_noisy_data, y_space))
    y_normalized_all = min_max_scaler.fit_transform(y_all)

    y_data_temp = y_normalized_all[:n_samples_combined, :]
    y_space_temp = y_normalized_all[n_samples_combined:, :]

    constant1 = max(y_data_temp)
    constant2 = max(y_space_temp)
    print(f'{constant1}, {constant2}')
    if constant1 <= constant2:
        num += 1
        interation = 0   #迭代次数
        gpr = GPR(kernel=ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=(1e-40,1e+5)),n_restarts_optimizer=10)
        gpr.fit(X_data_temp, y_data_temp)
        print(gpr.kernel_)
        regressor = GPR(kernel=gpr.kernel_, n_restarts_optimizer=2)
        while True:
            a, b, c, d, e, f = EI(regressor, X_data_temp, y_data_temp, X_space_temp, y_space_temp)
            interation += 1
            if b > constant1:
                X_data_temp = np.append(X_data_temp, a).reshape(-1, 6)
                y_data_temp = np.append(y_data_temp, b).reshape(-1, 1)
                X_space_temp = np.delete(X_space_temp, c, axis=0).reshape(-1, 6)
                y_space_temp = np.delete(y_space_temp, c).reshape(-1, 1)
                print(f"随机种子为{i},迭代{interation}次,已执行{num}次")
                interaions.append(interation)
                break
            else:
                X_data_temp = np.append(X_data_temp, a).reshape(-1, 6)
                y_data_temp = np.append(y_data_temp, b).reshape(-1, 1)
                X_space_temp = np.delete(X_space_temp, c, axis=0).reshape(-1, 6)
                y_space_temp = np.delete(y_space_temp, c).reshape(-1, 1)
        i += 1
    else:
        i += 1
df = pd.DataFrame(interaions)
# 指定输出的 Excel 文件路径
output_file_path = r'D:\课程资料\本科毕设\本科毕业设计\师兄小论文数据\low_noise.xlsx'
# 将 DataFrame 写入 Excel 文件
df.to_excel(output_file_path, index=False, sheet_name='Interactions')
