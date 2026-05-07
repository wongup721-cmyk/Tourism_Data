import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 生成示例数据
# 读取CSV文件
file_path = r'C:\Users\29324\Desktop\最终处理(最终版)\timenet +tsmixer\timenet\data\旅游人数.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(file_path)

# 假设数据在某一列中
data_column = 'N-tourist'  # 替换为你实际的列名
data = df[data_column].values

# 归一化数据
scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# 准备时间序列数据
def create_dataset(data, time_window):
    X, y = [], []
    for i in range(len(data) - time_window):
        X.append(data[i:i + time_window])
        y.append(data[i + time_window])
    return np.array(X), np.array(y)



time_window = 30
X, y = create_dataset(data, time_window)
# 划分训练集和测试集
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

# 重塑输入数据以适应CNN输入格式 (samples, time_steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 构建CNN模型
model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(time_window, 1)),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 训练模型
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))

# 进行预测
y_pred = model.predict(X_test)

# 反归一化数据
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred = scaler.inverse_transform(y_pred).flatten()

# 画出真实值和预测值对比
plt.figure(figsize=(14, 7))
plt.plot(y_test, label='True Value')
plt.plot(y_pred, label='Predicted Value')
plt.title('True vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.show()
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
# 计算评估指标
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test))

# 创建包含评估指标的DataFrame
metrics_df = pd.DataFrame({
    '评估指标': ['R²', 'MSE', 'RMSE', 'MAE', 'MAPE'],
    '值': [r2, mse, rmse, mae, mape]
})

# 设定保存路径
save_path = r'C:\Users\29324\Desktop\最终处理(最终版)\timenet +tsmixer\timenet\result\CNN'
import os

# 确保目录存在
os.makedirs(save_path, exist_ok=True)

# 保存评估指标到Excel文件
metrics_file_path = os.path.join(save_path, '旅游人数6.xlsx')
metrics_df.to_excel(metrics_file_path, index=False)

# 打印结果
print(f'R²: {r2:.4f}')
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.4f}')
print(f'\n评估指标已保存到: {metrics_file_path}')

# 将真实值和预测值合并为一个 DataFrame
result_df = pd.DataFrame({'真实值': y_test.flatten(), '预测值': y_pred.flatten()})
# 保存 DataFrame 到一个 CSV 文件
result_df.to_csv('真实值与预测值.csv', index=False, encoding='utf-8')
# 打印保存成功的消息
print('真实值和预测值已保存到真实值与预测值.csv文件中。')