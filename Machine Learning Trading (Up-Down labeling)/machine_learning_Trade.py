# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:13:12 7222

@author: user
"""

import glob
import os
import pandas as pd
import  talib
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


files=glob.glob('C:/Users/user/Desktop/대학원수업/변동성라벨링_SCI/나스닥100 종목/*.csv')

path = "C:/Users/user/Desktop/대학원수업/변동성라벨링_SCI/나스닥100 종목/"

file_list =os.listdir(path)

len(file_list)

df=[]

for file in file_list:
    path = "C:/Users/user/Desktop/대학원수업/변동성라벨링_SCI/나스닥100 종목/"
    data=pd.read_csv(path+"/"+file)
    data=data.drop(['Unnamed: 0'],axis=1)
    df.append(data)






for  i in df:
    ADX=talib.ADX(i.high,i.low,i.close,timeperiod=14)

    ADXR=talib.ADXR(i.high,i.low,i.close,timeperiod=14)
    
    APO=talib.APO(i.close,fastperiod=12,slowperiod=26,matype=0)
    
    aroondown,aroonup =talib.AROON(i.high, i.low, timeperiod=14)
    
    AROONOSC=talib.AROONOSC(i.high,i.low,timeperiod=14)
    
    BOP=talib.BOP(i.open,i.high,i.low,i.close)
    
    CCI=talib.CCI(i.high,i.low,i.close,timeperiod=14)
    
    CMO=talib.CMO(i.close,timeperiod=14)
    
    DX=talib.DX(i.high,i.low,i.close,timeperiod=14)
    
    macd, macdsignal, macdhist = talib.MACD(i.close, fastperiod=12, slowperiod=26, signalperiod=9)
    
    ma_macd, ma_macdsignal, ma_macdhist = talib.MACDEXT(i.close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    
    fix_macd,fix_macdsignal, fix_macdhist = talib.MACDFIX(i.close, signalperiod=9)
    
    MFI=talib.MFI(i.high, i.low,i.close, i.volume, timeperiod=14)
    
    MINUS_DI=talib.MINUS_DI(i.high, i.low, i.close, timeperiod=14)
    
    MINUS_DM=talib. MINUS_DM(i.high, i.low, timeperiod=14)
    
    MOM=talib.MOM(i.close,timeperiod=10)
    
    PLUS_DM=talib.PLUS_DM(i.high,i.low,timeperiod=14)
    
    PPO=talib.PPO(i.close, fastperiod=12, slowperiod=26, matype=0)
    
    ROC=talib.ROC(i.close,timeperiod=10)
    
    ROCP=talib.ROCP(i.close,timeperiod=10)
    
    ROCR=talib.ROCR(i.close,timeperiod=10)
    
    ROCR100=talib.ROCR100(i.close,timeperiod=10)
    
    RSI=talib.RSI(i.close,timeperiod=14)
    
    slowk, slowd = talib.STOCH(i.high, i.low, i.close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    
    fastk, fastd = talib.STOCHF(i.high, i.low, i.close, fastk_period=5, fastd_period=3, fastd_matype=0)
    
    TRIX=talib.TRIX(i.close,timeperiod=72)
    
    ULTOSC=talib.ULTOSC(i.high,i.low,i.close,timeperiod1=7,timeperiod2=14,timeperiod3=28)
    
    WILLR=talib.WILLR(i.high,i.low,i.close,timeperiod=14)
    
    
    i['ADX']=ADX
    i['ADXR']=ADXR
    i['APO']=APO
    i['aroondown']=aroondown
    i['aroonup']=aroonup
    i['AROONOSC']=AROONOSC
    i['BOP']=BOP
    i['CCI']=CCI
    i['CMO']=CMO
    i['DX']=DX
    i['MACD']=macd
    i['macdsignal']=macdsignal
    i['macdhist']=macdhist
    i['ma_macd']=ma_macd
    i['ma_macdsignal']=ma_macdsignal
    i['ma_macdhist']=ma_macdhist
    i['fix_macd']=fix_macd
    i['fix_macdsignal']=fix_macdsignal
    i['fix_macdhist']=fix_macdhist
    i['MFI']=MFI
    i['MINUS_DI']=MINUS_DI
    i['MINUS_DM']=MINUS_DM
    i['MOM']=MOM
    i['PLUS_DM']=PLUS_DM
    i['PPO']=PPO
    i['ROC']=ROC
    i['ROCP']=ROCP
    i['ROCR']=ROCR
    i['ROCR100']=ROCR100
    i['RSI']=RSI
    i['slowk']=slowk
    i['slowd']=slowd
    i['fastk']=fastk
    i['fastd']=fastd
    i['TRIX']=TRIX
    i['ULTOSC']=ULTOSC
    i['WILLR']=WILLR
    

#########7209~7272년만 뽑기




#####
for i in df:
    i['diff']=i.close.diff().shift(-1).fillna(0)
    i['Label'] = None
    

#label 생성

for i in  range(0,72):
    for e in df[i].index:
        if df[i]['diff'][e] > 0:
            df[i]['Label'][e] = '1'
        elif df[i]['diff'][e]==0:
            df[i]['Label'][e] ='0'
        else:        
            df[i]['Label'][e] = '0'



#인덱스 번호 한번더 초기화
for i in range(0,72):
    df[i]=df[i].reset_index(drop=True)
    df[i]=df[i].fillna(0)






###########modeling

#model train/test set 생성  
train_data=[]
test_data=[]

############train/test 분리 ###
for i in range(0,72):
    train=None
    train=df[i]['date'].str.contains('2014|2015|2016|2017|2018')
    train_data.append(df[i][train])
for i in range(0,72):
    test=None    
    test=df[i]['date'].str.contains('2019|2020')
    test_data.append(df[i][test])
    

for i in range(0,72):
    train_data[i]=train_data[i].drop(['date','open','high','low','close','volume','change','diff'],axis=1)
    test_data[i]=test_data[i].drop(['date','open','high','low','close','volume','change','diff'],axis=1)
    


for i in range(0,72):
    print(df[i].isnull().sum())

#x_train,y_train,x_test,y_test

x_train =[]
y_train =[]
x_test=[]
y_test=[]


#######7216
for i in range(0,72):
    x_train.append(train_data[i].drop(['Label'],axis=1))
    y_train.append(train_data[i]['Label'])
    
    x_test.append(test_data[i].drop(['Label'],axis=1))
    y_test.append(test_data[i]['Label']) 

    
#모델링
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier



    
#모델링
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier



pred=[]
pred_decision=[]
pred_naive=[]
pred_randomforest=[]
pred_knn=[]
pred_neural=[]
pred_voting=[]
pred_gbm=[]
pred_xgb=[]


random_state = 0

for i in range(0,72):
    #logistic
    logistic =LogisticRegression(random_state= random_state)
    logistic.fit(x_train[i],y_train[i])
    
    pred.append(logistic.predict(x_test[i]))

    
    ##############decision tree
    dt=DecisionTreeClassifier(random_state= random_state)
    
    dt.fit(x_train[i],y_train[i])
    pred_decision.append(dt.predict(x_test[i]))
    
    
    ##############naive
    naive=GaussianNB()
    
    naive.fit(x_train[i],y_train[i])
    
    pred_naive.append(naive.predict(x_test[i]))
    
    
    #############randomforest
    randomforest=RandomForestClassifier(random_state= random_state)
    
    randomforest.fit(x_train[i],y_train[i])
    
    pred_randomforest.append(randomforest.predict(x_test[i]))
    

    ###############knn
    knn=KNeighborsClassifier(n_neighbors=3)
    
    knn.fit(x_train[i],y_train[i])
    
    pred_knn.append(knn.predict(x_test[i]))
    
    
    ###############nueral
    
    nueral=MLPClassifier(random_state=random_state)
    
    nueral.fit(x_train[i],y_train[i])
    
    pred_neural.append(nueral.predict(x_test[i]))
    
    
    ###########voting
    
    voting=VotingClassifier(estimators=[('decison',dt),('knn',knn),('logisitc',logistic),
                                        ('naive',naive),('nueral',nueral)],voting='hard')
    
    voting.fit(x_train[i],y_train[i])
    
    pred_voting.append(voting.predict(x_test[i]))
     
    ########gbm
    gbm=GradientBoostingClassifier(random_state=random_state)
    
    gbm.fit(x_train[i],y_train[i])
    
    pred_gbm.append(gbm.predict(x_test[i]))
    
    ########XGBoost
    xgb=XGBClassifier(random_state=random_state)
    
    xgb.fit(x_train[i],y_train[i])
    
    pred_xgb.append(xgb.predict(x_test[i]))
    


# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:31:36 2022

@author: user
"""

test_7219=[]


for i in range(0,72):
    test=None    
    test=df[i]['date'].str.contains('2019|2020')
    test_7219.append(df[i][test])

for i in range(0,72):
    test_7219[i]['pred']=pred[i]
    test_7219[i]['pred_decision']=pred_decision[i]
    test_7219[i]['pred_naive']=pred_naive[i]
    test_7219[i]['pred_randomforest']=pred_randomforest[i]
    test_7219[i]['pred_knn']=pred_knn[i]
    test_7219[i]['pred_neural']=pred_neural[i]
    test_7219[i]['pred_voting']=pred_voting[i]
    test_7219[i]['pred_gbm']=pred_gbm[i]

#pred 자료형 변경
for i in range(0,72):
    test_7219[i]['pred']=test_7219[i]['pred'].astype('float')
    test_7219[i]['pred_decision']=test_7219[i]['pred_decision'].astype('float')
    test_7219[i]['pred_naive']=test_7219[i]['pred_naive'].astype('float')
    test_7219[i]['pred_randomforest']=test_7219[i]['pred_randomforest'].astype('float')
    test_7219[i]['pred_knn']=test_7219[i]['pred_knn'].astype('float')
    test_7219[i]['pred_neural']=test_7219[i]['pred_neural'].astype('float')
    test_7219[i]['pred_voting']=test_7219[i]['pred_voting'].astype('float')
    test_7219[i]['pred_gbm']=test_7219[i]['pred_gbm'].astype('float')
    
    

#새로운 라벨 추가(e -> 인덱스 번호)
for i in range(0,72):
    test_7219[i]['position']=None
    
                       
#randomforest
for i in range(0,72):
    for e in test_7219[i].index:
        try:
            if test_7219[i]['pred_randomforest'][e]+test_7219[i]['pred_randomforest'][e+1]==0:
                test_7219[i]['position'][e+1]='no action'
            elif test_7219[i]['pred_randomforest'][e]+test_7219[i]['pred_randomforest'][e+1]==2:
                test_7219[i]['position'][e+1]='holding'
            elif test_7219[i]['pred_randomforest'][e] > test_7219[i]['pred_randomforest'][e+1]:
                test_7219[i]['position'][e+1]='sell'
            else:
                test_7219[i]['position'][e+1]='buy'
        except:
            pass

#첫날 position이 holding일 경우 buy로 변경
for i in range(0,72):
    if test_7219[i]['position'][test_7219[i].index[0]]=='holding':
        test_7219[i]['position'][test_7219[i].index[0]]='buy'
    elif test_7219[i]['position'][test_7219[i].index[0]]=='sell':
        test_7219[i]['position'][test_7219[i].index[0]]='buy'
    else:
        test_7219[i]['position'][test_7219[i].index[0]]='buy'

# for i in range(len(test_7219)):
#     print(test_7219[i]['position'][1472])
#     print(test_7219[i]['position'][0])
#     print(test_7219[i]['position'].value_counts())

#강제 청산
for i in range(0,72):
    for e in test_7219[i].index[-1:]:
        if test_7219[i]['position'][e]=='holding':
            test_7219[i]['position'][e]='sell'
        elif test_7219[i]['position'][e]=='buy':
            test_7219[i]['position'][e]='sell'
        elif test_7219[i]['position'][e]=='no action':
            test_7219[i]['position'][e]='sell'
        else:
            print(i)



for i in range(0,72):
    test_7219[i]['profit']=None
    
#다음날 시가를 가져오게 생성
for i in range(0,72):
    for e in test_7219[i].index:
        try:
            if test_7219[i]['position'][e]=='buy':
                test_7219[i]['profit'][e]=test_7219[i]['open'][e+1]
            elif test_7219[i]['position'][e]=='sell':
                test_7219[i]['profit'][e]=test_7219[i]['open'][e+1]
            else:
                print(i)
        except:
            pass



for i in range(0,72):
    for e in test_7219[i].index[-1:]:
        if test_7219[i]['position'][e]=='sell':
            test_7219[i]['profit'][e]=test_7219[i]['open'][e]
        
####

buy_label=[]
for i in range(0,72):
    buy_position=test_7219[i]['position']=='buy'
    buy_label.append(test_7219[i][buy_position])
    
sell_label=[]
for i in range(0,72):
    sell_position=test_7219[i]['position']=='sell'
    sell_label.append(test_7219[i][sell_position])    


buy=[]
sell=[]
for i in range(0,72):
    buy.append(buy_label[i]['open'].reset_index(drop=True))
    sell.append(sell_label[i]['open'].reset_index(drop=True))
    
  
profit_2=[]    
for i in range(0,72):
    profit_2.append((sell[i]-(0.0015*sell[i]))-buy[i])
  

for i in range(0,72):
    test_7219[i]['profit_2']=None
    

#profit 결측치 처리
for i in range(0,72):
    profit_2[i]=profit_2[i].dropna()
    
    
#profit_2 sell에 해당하는 행에 값 넣기
for tb, pf in zip(test_7219, profit_2):
    total_idx = tb[tb['position'] == 'sell'].index
    total_pf_idx = pf.index
    for idx, pf_idx in zip(total_idx, total_pf_idx):
        tb.loc[idx, 'profit_2'] = pf[pf_idx]




for i in range(0,72):
    test_7219[i]['profit_cumsum']=None
    
    
    

#profit 누적 합 
for i in range(0,72):
    for e in test_7219[i].index:
        try:
            if test_7219[i]['position'][e]=='holding':
                test_7219[i]['profit_2'][e]=0
            elif test_7219[i]['position'][e]=='no action':
                test_7219[i]['profit_2'][e]=0
            elif test_7219[i]['position'][e]=='buy':
                test_7219[i]['profit_2'][e]=0
            else:
                print(i)
        except:
            pass


#새로운 청산 기준 누적합

for i in range(0,72):
    test_7219[i]['profit_cumsum2']=None    
    
    
for i in range(0,72):
    test_7219[i]['profit_cumsum']=test_7219[i]['profit_2'].cumsum()



################# ratio 작성

#ratio 작성
for i in range(0,72):
    profit_2[i]=pd.DataFrame(profit_2[i])

#거래횟수
trade= []

for i in range(0,72):
    trade.append(len(profit_2[i]))
    
#승률


for i in range(0,72):
    profit_2[i]['average']=None

   
for i in range(0,72):
    for e in range(len(profit_2[i])):      
        if profit_2[i]['open'][e] > 0:
            profit_2[i]['average'][e]='gain'
        else:
            profit_2[i]['average'][e]='loss'
            
for i in range(0,72):
    for e in range(len(profit_2[i])):
        if profit_2[i]['open'][e] < 0:
            profit_2[i]['open'][e]=profit_2[i]['open'][e] * -1
        else:
            print(i)

win=[]
for i in range(0,72):
    try:
        win.append(profit_2[i].groupby('average').size()[0]/len(profit_2[i]))
    except:
        win.append('0')
    
#평균 수익

gain=[]

for i in range(0,72):
    gain.append(profit_2[i].groupby('average').mean())
    

real_gain=[]

for i in range(0,72):
    try:
        real_gain.append(gain[i]['open'][0])
    except:
        real_gain.append('0')



#평균 손실
loss=[]

for i in range(0,72):
    try:
        loss.append(gain[i]['open'][1])
    except:
        loss.append('0')

    
loss
#payoff ratio
payoff=[]

for i in range(0,72):
    try:
        payoff.append(gain[i]['open'][0]/gain[i]['open'][1])
    except:
        payoff.append('inf')
    
#profit factor

factor_sum=[]

len(factor_sum)
for i in range(0,72):
    factor_sum.append(profit_2[i].groupby('average').sum())

factor=[]

for i in range(0,72):
    try:
        factor.append(factor_sum[i]['open'][0]/factor_sum[i]['open'][1])
    except:
        factor.append('0')

#year
year=[]

for i in range(0,72):
    year.append('2019~2020')

#최종 결과물 파일 작성
file_name = file_list

stock_name=pd.DataFrame({'stock_name':file_name})

year=pd.DataFrame({'year':year})

trade=pd.DataFrame({'No.trades':trade})

win=pd.DataFrame({'Win%':win})

real_gain=pd.DataFrame({'Average gain($)':real_gain})

loss=pd.DataFrame({'Average loss($)':loss})

payoff=pd.DataFrame({'Payoff ratio':payoff})

factor=pd.DataFrame({'Profit factor':factor})

#7272
result =pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)


model_name = ['pred_logistic','pred_decision','pred_naive','pred_randomforest','pred_knn','pred_neural','pred_voting','pred_gbm']

result.to_csv('C:/Users/user/Desktop/강화학습_sample_trading/train_set_result/test_result_{}.csv'.format(model_name[7]),encoding='euc-kr')
