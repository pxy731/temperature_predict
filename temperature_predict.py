
# 回归预测温度
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# 使用keras建模方法
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

#（1）数据获取
filepath = 'D:\\computer\\python文件\\温度\\temps.csv'
features = pd.read_csv(filepath)
# tenmp2代表前两天的温度，temp1代表前一天的温度，目标值为actual

#（2）数据预处理
# ==1== 处理时间数据，将年月日组合在一起
import datetime
# 获取年月日数据
years = features['year']
months = features['month']
days = features['day']
# 将年月日拼接在一起--字符串类型
dates = []  
for year,month,day in zip(years,months,days):
    date = str(year)+'-'+str(month)+'-'+str(day)
    dates.append(date)
# 转变成datetime格式
times = []
for date in dates:
    time = datetime.datetime.strptime(date,'%Y-%m-%d')
    times.append(time)
# 看一下前5行
times[:5]

#（3）可视化，对各个特征绘图
# 指定绘图风格
plt.style.use('fivethirtyeight')
# 设置画布，2行2列的画图窗口，第一行画ax1和ax2
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(20,10))

# ==1== actual特征列
ax1.plot(times,features['actual'])
# 设置x轴y轴标签和title标题
ax1.set_xlabel('');ax1.set_ylabel('Temperature');ax1.set_title('actual temp')
# ==2== 前一天的温度
ax2.plot(times,features['temp_1'])
# 设置x轴y轴标签和title标题
ax2.set_xlabel('');ax2.set_ylabel('Temperature');ax2.set_title('temp_1')
# ==3== 前2天的温度
ax3.plot(times,features['temp_2'])
# 设置x轴y轴标签和title标题
ax3.set_xlabel('Date');ax3.set_ylabel('Temperature');ax3.set_title('temp_2')
# ==4== friend
ax4.plot(times,features['friend'])
# 设置x轴y轴标签和title标题
ax4.set_xlabel('Date');ax4.set_ylabel('Temperature');ax4.set_title('friend')
# 轻量化布局调整绘图
plt.tight_layout(pad=2)

#（4）对字符型数据one-hot编码
# week列是字符串，重新编码，变成数值型
features = pd.get_dummies(features)

#（5）划分特征值和目标值
# 获取目标值y，从Series类型变成数值类型
targets = np.array(features['actual'])
# 获取特征值x，即在原数据中去掉目标值列，默认删除行，需要指定轴axis=1指向列
features = features.drop('actual',axis=1)
# 把features从DateFrame变成数组
features = np.array(features)

#（6）标准化处理
from sklearn import preprocessing
input_features = preprocessing.StandardScaler().fit_transform(features)

#（7）keras构建网络模型
# ==1== 构建层次
model = tf.keras.Sequential()
# 隐含层1设置16层，权重初始化方法设置为随机高斯分布
# 加入正则化惩罚项
model.add(layers.Dense(16,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(layers.Dense(32,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(layers.Dense(1,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# ==2== 指定优化器
model.compile(optimizer=tf.keras.optimizers.SGD(0.001),loss='mean_squared_error')
# ==3== 网络训练
model.fit(input_features,targets,validation_split=0.25,epochs=100,batch_size=128)
# ==4== 网络模型结构
model.summary()
# ==5== 预测模型结果
predict = model.predict(input_features)

#（7）展示预测结果
# 真实值，蓝色实现
fig = plt.figure(figsize=(10,5))
axes = fig.add_subplot(111)
axes.plot(dates,targets,'bo',label='actual')
# 预测值，红色散点
axes.plot(dates,predict,'ro',label='predict')
axes.set_xticks(dates[::50])
axes.set_xticklabels(dates[::50],rotation=45)

plt.legend()
plt.show()
