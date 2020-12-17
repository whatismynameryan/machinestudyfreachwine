#coding:gbk
"""
程序目标：利用决策树算法进行分类
作者：李浩
日期：2020.12.17
"""
import pandas as pd           # 调入需要用的库
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sb

# 调入数据
df=pd.read_csv('frenchwine.csv')
df.columns=['species','alcohol', 'malic_acid', 'ash', 'alcalinity ash','magnesium']
# 查看前5条数据
print("试查看前五条数据：")
df.head()
print(df.head()) 

print("葡萄各参数指标的描述性统计结果:")
# 查看数据描述性统计信息
df.describe()
print(df.describe())

def scatter_plot_by_category(feat, x, y): #数据的可视化 
    alpha=0.5
    gs=df.groupby(feat)
    cs=cm.rainbow(np.linspace(0, 1, len(gs)))
    for g, c in zip(gs, cs):
        plt.scatter(g[1][x],g[1][y],color=c,alpha=alpha)

plt.figure(figsize=(20, 10)) #图1
for column_index, column in enumerate(df.columns[1:2]):
    if column=='species':
        continue
    plt.subplot(3,2, column_index+1)
    sb.violinplot(x='species', y=column, data=df)
plt.show()
plt.figure(figsize=(20, 10)) #图2
for column_index, column in enumerate(df.columns[2:3]):
    if column=='species':
        continue
    plt.subplot(3,2, column_index+1)
    sb.violinplot(x='species', y=column, data=df)
plt.show()
plt.figure(figsize=(20, 10)) #图3
for column_index, column in enumerate(df.columns[3:4]):
    if column=='species':
        continue
    plt.subplot(3,2, column_index+1)
    sb.violinplot(x='species', y=column, data=df)
plt.show()
plt.figure(figsize=(20, 10)) #图4
for column_index, column in enumerate(df.columns[4:5]):
    if column=='species':
        continue
    plt.subplot(3,2, column_index+1)
    sb.violinplot(x='species', y=column, data=df)
plt.show()
plt.figure(figsize=(20, 10)) #图5
for column_index, column in enumerate(df.columns[1:]):
    if column=='species':
        continue
    plt.subplot(3,2, column_index+1)
    sb.violinplot(x='species', y=column, data=df)
plt.show()
plt.figure(figsize=(20, 10)) #5张图汇总
for column_index, column in enumerate(df.columns[5:6]):
    if column=='species':
        continue
    plt.subplot(3,2, column_index+1)
    sb.violinplot(x='species', y=column, data=df)
plt.show()

# 首先对数据进行切分，即划分出训练集和测试集
from sklearn.model_selection import train_test_split #调入sklearn库中交叉检验，划分训练集和测试集
all_inputs = df[['alcohol', 'malic_acid', 'ash', 'alcalinity ash','magnesium']].values
all_species = df['species'].values

(X_train,
 X_test,
 Y_train,
 Y_test) = train_test_split(all_inputs, all_species, train_size=0.7, random_state=1)#70%的数据选为训练集
 
 


# 使用决策树算法进行训练
from sklearn.tree import DecisionTreeClassifier #调入sklearn库中的DecisionTreeClassifier来构建决策树
# 定义一个决策树对象
decision_tree_classifier = DecisionTreeClassifier()

# 训练模型
model = decision_tree_classifier.fit(X_train, Y_train)
# 输出模型的准确度
print("模型的准确度为:")#文字提示
print(decision_tree_classifier.score(X_test, Y_test))



def English_to_Chinese(number):    #自定义函数实现英文转为中文。
    if number=="Zinfandel":
        return "仙粉黛"
    elif number=="Sauvignon":
        return "赤霞珠"
    elif number=="Syrah":
        return "西拉"
print("X_test中的全部数据分类的结果:")#汉字提醒
model.predict(X_test[:])
count=0
for number in model.predict(X_test[:]):
    number1=English_to_Chinese(number) # 输出测试的结果的具体中文，即输出模型预测的结果
    count=count+1
    print(number1,end=" ") #调整排版
    if count%5==0:
        print("")
print(" ")
print("----------------------")

print("对已知的三个数据进行分类的结果分别为:")  #对已知的三个数据进行分类
data1=model.predict([[13.42,3.21,2.62,23.5,95]])
data2=model.predict([[ 12.32,2.77,2.37,22,90]])
data3=model.predict([[13.75,1.59,2.7,19.5,135]])
data4=English_to_Chinese(data1)
data5=English_to_Chinese(data2)
data6=English_to_Chinese(data3)
print(data4,data5,data6,end=" ")#输出并调整排版
print(" ")
print("------------------")

##决策树可视化
from IPython.display import Image  
#from sklearn.externals.six import StringIO  #sklearn 0.23版本已经删掉了这个包,直接安装six即可
from six import StringIO
from sklearn.tree import export_graphviz
features = list(df.columns[:-1])
print(features)
import pydotplus
import os #要安装一个Graphviz软件
os.environ['PATH'] = os.environ['PATH'] + (';c:\\Program Files\\Graphviz\\bin\\') #
dot_data=StringIO()
export_graphviz(decision_tree_classifier, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph[0].create_png())  
graph.write_pdf("frenchwine.pdf") #将iris数据集利用决策树算法可视化结果保持到frenchwine.pdf文件中


