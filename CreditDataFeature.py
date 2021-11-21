from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import missingno
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('credit.csv', header = 'infer', index_col= None)
missing = pd.DataFrame(df.isnull().sum()/df.shape[0],columns=['missing_rate']).reset_index()
missing =  missing.sort_values(by='missing_rate', ascending=False)[:15]

miss_name = []
for i in df:
    if df[i].isnull().sum() > 0 :
        miss_name.append(i)
for i in miss_name:
    df[i].fillna(int(df[i].mode()))

cross_table=pd.crosstab(df.House_State, columns=df.Target, margins=True)
cross_table_rowpct=cross_table.div(cross_table['All'],axis=0)

temp=[]
for i in df:
    print(i, df[i].value_counts().count())
    if df[i].value_counts().count() < 10:
        temp.append(i)
temp = df[temp]

temp = temp.drop('Target', axis=1)
y = df['Target']
for i in temp:
    temp[i]=temp[i].fillna(int(temp[i].mode()))
(chi2,pval)=chi2(temp,y)

dict_feature={}
for i,j in zip(temp,chi2):
    dict_feature[i]=j
end=sorted(dict_feature.items(),key=lambda item:item[1],reverse=True)
lianxu=[i for i in df if i not in end]
temp=[]
for i in range(len(end)):
    print(end[i][1])
    if end[i][1] < 1 :
        temp.append(end[i])
print(temp)
for i in temp:
    end.remove(i)
classification = [i[0] for i in end]

lianxu=[i for i in df if i not in end]
lianxu=df[lianxu]
lianxu=lianxu.drop(['Cust_No','Target'],axis=1)
corr=lianxu.corr()
plt.figure(figsize=(25,15))

sns.heatmap(corr,annot=True)
end = []
for index_ in corr.index:
    for col_ in corr.columns:
        if corr.loc[index_,col_] >= 0.8 and index_!=col_ and (col_,index_) not in end:
            end.append((index_,col_))
reg=[i[1] for i in end]

X = df.drop('Target',axis=1)
y = df['Target']

# print(np.isnan(X).any())
X.fillna(method='ffill', inplace=True)

x_rfe = RFE(estimator=LogisticRegression(), n_features_to_select=20).fit(X, y)
print(x_rfe.n_features_ )
print(x_rfe.support_ )
print(x_rfe.ranking_ )
print(x_rfe.estimator_ )

name=[]
for i,j in zip(x_rfe.support_,X):
    print(i,j)
    if i == True :
        name.append(j)
print(name)

print(x_rfe.n_features_ )
print(x_rfe.support_ )
print(x_rfe.ranking_ )
print(x_rfe.estimator_ )

data=X[name]
data.to_csv('good_feaute.csv')



