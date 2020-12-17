#coding:gbk
"""
����Ŀ�꣺���þ������㷨���з���
���ߣ����
���ڣ�2020.12.17
"""
import pandas as pd           # ������Ҫ�õĿ�
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sb

# ��������
df=pd.read_csv('frenchwine.csv')
df.columns=['species','alcohol', 'malic_acid', 'ash', 'alcalinity ash','magnesium']
# �鿴ǰ5������
print("�Բ鿴ǰ�������ݣ�")
df.head()
print(df.head()) 

print("���Ѹ�����ָ���������ͳ�ƽ��:")
# �鿴����������ͳ����Ϣ
df.describe()
print(df.describe())

def scatter_plot_by_category(feat, x, y): #���ݵĿ��ӻ� 
    alpha=0.5
    gs=df.groupby(feat)
    cs=cm.rainbow(np.linspace(0, 1, len(gs)))
    for g, c in zip(gs, cs):
        plt.scatter(g[1][x],g[1][y],color=c,alpha=alpha)

plt.figure(figsize=(20, 10)) #ͼ1
for column_index, column in enumerate(df.columns[1:2]):
    if column=='species':
        continue
    plt.subplot(3,2, column_index+1)
    sb.violinplot(x='species', y=column, data=df)
plt.show()
plt.figure(figsize=(20, 10)) #ͼ2
for column_index, column in enumerate(df.columns[2:3]):
    if column=='species':
        continue
    plt.subplot(3,2, column_index+1)
    sb.violinplot(x='species', y=column, data=df)
plt.show()
plt.figure(figsize=(20, 10)) #ͼ3
for column_index, column in enumerate(df.columns[3:4]):
    if column=='species':
        continue
    plt.subplot(3,2, column_index+1)
    sb.violinplot(x='species', y=column, data=df)
plt.show()
plt.figure(figsize=(20, 10)) #ͼ4
for column_index, column in enumerate(df.columns[4:5]):
    if column=='species':
        continue
    plt.subplot(3,2, column_index+1)
    sb.violinplot(x='species', y=column, data=df)
plt.show()
plt.figure(figsize=(20, 10)) #ͼ5
for column_index, column in enumerate(df.columns[1:]):
    if column=='species':
        continue
    plt.subplot(3,2, column_index+1)
    sb.violinplot(x='species', y=column, data=df)
plt.show()
plt.figure(figsize=(20, 10)) #5��ͼ����
for column_index, column in enumerate(df.columns[5:6]):
    if column=='species':
        continue
    plt.subplot(3,2, column_index+1)
    sb.violinplot(x='species', y=column, data=df)
plt.show()

# ���ȶ����ݽ����з֣������ֳ�ѵ�����Ͳ��Լ�
from sklearn.model_selection import train_test_split #����sklearn���н�����飬����ѵ�����Ͳ��Լ�
all_inputs = df[['alcohol', 'malic_acid', 'ash', 'alcalinity ash','magnesium']].values
all_species = df['species'].values

(X_train,
 X_test,
 Y_train,
 Y_test) = train_test_split(all_inputs, all_species, train_size=0.7, random_state=1)#70%������ѡΪѵ����
 
 


# ʹ�þ������㷨����ѵ��
from sklearn.tree import DecisionTreeClassifier #����sklearn���е�DecisionTreeClassifier������������
# ����һ������������
decision_tree_classifier = DecisionTreeClassifier()

# ѵ��ģ��
model = decision_tree_classifier.fit(X_train, Y_train)
# ���ģ�͵�׼ȷ��
print("ģ�͵�׼ȷ��Ϊ:")#������ʾ
print(decision_tree_classifier.score(X_test, Y_test))



def English_to_Chinese(number):    #�Զ��庯��ʵ��Ӣ��תΪ���ġ�
    if number=="Zinfandel":
        return "�ɷ���"
    elif number=="Sauvignon":
        return "��ϼ��"
    elif number=="Syrah":
        return "����"
print("X_test�е�ȫ�����ݷ���Ľ��:")#��������
model.predict(X_test[:])
count=0
for number in model.predict(X_test[:]):
    number1=English_to_Chinese(number) # ������ԵĽ���ľ������ģ������ģ��Ԥ��Ľ��
    count=count+1
    print(number1,end=" ") #�����Ű�
    if count%5==0:
        print("")
print(" ")
print("----------------------")

print("����֪���������ݽ��з���Ľ���ֱ�Ϊ:")  #����֪���������ݽ��з���
data1=model.predict([[13.42,3.21,2.62,23.5,95]])
data2=model.predict([[ 12.32,2.77,2.37,22,90]])
data3=model.predict([[13.75,1.59,2.7,19.5,135]])
data4=English_to_Chinese(data1)
data5=English_to_Chinese(data2)
data6=English_to_Chinese(data3)
print(data4,data5,data6,end=" ")#����������Ű�
print(" ")
print("------------------")

##���������ӻ�
from IPython.display import Image  
#from sklearn.externals.six import StringIO  #sklearn 0.23�汾�Ѿ�ɾ���������,ֱ�Ӱ�װsix����
from six import StringIO
from sklearn.tree import export_graphviz
features = list(df.columns[:-1])
print(features)
import pydotplus
import os #Ҫ��װһ��Graphviz���
os.environ['PATH'] = os.environ['PATH'] + (';c:\\Program Files\\Graphviz\\bin\\') #
dot_data=StringIO()
export_graphviz(decision_tree_classifier, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph[0].create_png())  
graph.write_pdf("frenchwine.pdf") #��iris���ݼ����þ������㷨���ӻ�������ֵ�frenchwine.pdf�ļ���


