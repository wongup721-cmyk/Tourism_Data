import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# 修改数据读取部分
try:
    # 读取xlsx文件
    data = pd.read_excel(r'C:\Users\29324\Desktop\最终处理(最终版)\timenet +tsmixer\timenet\data\agent剔除疫情.xlsx')
    # 将date列设置为索引
    if 'date' in data.columns:
        data.set_index('date', inplace=True)
    
    # 选择特定的列进行预测
    selected_features = ['N-tourist']
    target = 'N-tourist'
    
    # 确保所有需要的列都存在
    if not all(col in data.columns for col in selected_features + [target]):
        missing_cols = [col for col in selected_features + [target] if col not in data.columns]
        raise ValueError(f"数据集中缺少以下列: {missing_cols}")
    
    # 选择特征和目标变量
    X_data = data[selected_features]
    y_data = data[[target]]
    
    # 合并特征和目标变量用于训练
    data = pd.concat([X_data, y_data], axis=1)
    
    print("选择的特征:", selected_features)
    print("预测目标:", target)
    print("数据的形状:", data.shape)
    
except FileNotFoundError:
    print("错误：找不到文件，请检查文件路径是否正确")
    exit(1)
except Exception as e:
    print(f"读取文件时发生错误: {str(e)}")
    exit(1)

# 修改数据预处理部分
# 分别对特征和目标进行归一化
sc_X = MinMaxScaler()
sc_y = MinMaxScaler()

# 分别进行归一化
X_scaled = sc_X.fit_transform(X_data)
y_scaled = sc_y.fit_transform(y_data)

# 合并数据用于滑动窗口
data_scaled = np.column_stack((X_scaled, y_scaled))

# 修改数据处理部分
def sliding_windows(data, seq_length):
    x = []
    y = []
    for i in range(len(data)-seq_length):
        _x = data[i:(i+seq_length), :-1]  # 只取特征列
        _y = data[i+seq_length, -1:]      # 只取目标列（旅游人数）
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)

sc = MinMaxScaler()
data = sc.fit_transform(data)

seq_length = 4
x, y = sliding_windows(data, seq_length)

train_size = int(len(y) * 0.7)
test_size = len(y) - train_size

dataX = torch.Tensor(x).cuda()  # shape: (samples, seq_length, features)
dataY = torch.Tensor(y).cuda()  # shape: (samples, features)

trainX = torch.Tensor(x[0:train_size]).cuda()
trainY = torch.Tensor(y[0:train_size]).cuda()

testX = torch.Tensor(x[train_size:len(x)]).cuda()
testY = torch.Tensor(y[train_size:len(y)]).cuda()

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        batch_size = x.size(0)
        
        h_0 = Variable(torch.zeros(
            self.num_layers, batch_size, self.hidden_size)).to(x.device)
        c_0 = Variable(torch.zeros(
            self.num_layers, batch_size, self.hidden_size)).to(x.device)
        
        # LSTM forward
        out, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        # 只使用最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

num_epochs = 100
learning_rate = 0.01
input_size = len(selected_features)  # 输入维度为特征数量
hidden_size = 32  # 增加隐藏层大小以处理更多特征
num_layers = 2   # 增加LSTM层数以提高模型容量
num_classes = 1  # 输出维度改为1，只预测旅游人数

lstm = LSTM(num_classes, input_size, hidden_size, num_layers).cuda()

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)


# 在训练循环之前添加这些打印语句
print(f"训练数据 X 的维度: {trainX.shape}")
print(f"训练数据 Y 的维度: {trainY.shape}")
print(f"测试数据 X 的维度: {testX.shape}")
print(f"测试数据 Y 的维度: {testY.shape}")

# Train the model
lstm.train()  # 设置为训练模式
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = lstm(trainX)
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:  # 每10个epoch打印一次
        print(f"Epoch: {epoch}, Loss: {loss.item():.6f}")

lstm.eval()
test_predict = lstm(testX)

# 修改预测结果的处理部分
data_predict = test_predict.cpu().data.numpy()
dataY_plot = testY.cpu().data.numpy()

# 使用y的缩放器进行反向转换
predict = sc_y.inverse_transform(data_predict)
M = sc_y.inverse_transform(dataY_plot)

# 绘图部分
plt.figure(figsize=(12, 6))
plt.plot(M, c='k', label='实际值')
plt.plot(predict, c='r', label='预测值')
plt.legend()
plt.title('澳门旅游人数预测结果')
plt.xlabel('时间步')
plt.ylabel('旅游人数')
plt.grid(True)
plt.show()

# 计算平均绝对百分比误差（Mean Absolute Percentage Error，MAPE）
mape = mean_absolute_percentage_error(M, predict)
# 打印 MAPE 的值，保留三位小数
print('MAPE %.3f' %(mape))

# 计算均方根误差（Root Mean Squared Error，RMSE）
RMSE = math.sqrt(mean_squared_error(M, predict))
# 打印 RMSE 的值，保留三位小数
print('RMSE %.3f' %(RMSE))

# 计算平均绝对误差（Mean Absolute Error，MAE）
MAE = mean_absolute_error(M, predict)
# 打印 MAE 的值，保留三位小数
print('MAE %.3f' %(MAE))

# 计算确定系数（Coefficient of Determination，R2）
R2 = r2_score(M, predict)
# 打印 R2 的值，保留三位小数
print('R2 %.3f' %(R2))

# 创建保存结果的目录
import os
save_dir = r'C:\Users\29324\Desktop\最终处理(最终版)\timenet +tsmixer\timenet\result\lstm'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 创建结果数据框
results_df = pd.DataFrame({
    '评估指标': ['MAPE', 'RMSE', 'MAE', 'R2'],
    '数值': [mape, RMSE, MAE, R2]
})

# 获取用户输入的文件名
custom_filename = input("测试2（不需要包含.xlsx扩展名）：")
excel_filename = os.path.join(save_dir, f'{custom_filename}.xlsx')

# 保存结果到Excel文件
results_df.to_excel(excel_filename, index=False)
print(f"\n评估结果已保存到: {excel_filename}")

# 可选：同时保存使用的特征信息
feature_info = pd.DataFrame({
    '类型': ['选择的特征'] * len(selected_features) + ['预测目标'],
    '变量名': selected_features + [target]
})
with pd.ExcelWriter(excel_filename, mode='a') as writer:
    feature_info.to_excel(writer, sheet_name='特征信息', index=False)


# In[ ]




