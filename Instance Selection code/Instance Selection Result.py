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
    test_7219[i]['pred_xgb']=pred_xgb[i]

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
    test_7219[i]['pred_xgb']=test_7219[i]['pred_xgb'].astype('float')
    
    

#새로운 라벨 추가(e -> 인덱스 번호)
for i in range(0,72):
    test_7219[i]['position']=None
    
                       
#randomforest
for i in range(0,72):
    for e in test_7219[i].index:
        try:
            if test_7219[i]['pred'][e]+test_7219[i]['pred'][e+1]==0:
                test_7219[i]['position'][e+1]='no action'
            elif test_7219[i]['pred'][e]+test_7219[i]['pred'][e+1]==2:
                test_7219[i]['position'][e+1]='holding'
            elif test_7219[i]['pred'][e] > test_7219[i]['pred'][e+1]:
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


stock_name=pd.DataFrame({'stock_name':file_list})


for i in range(len(test_data)):
    stock_name['stock_name'][i] =stock_name['stock_name'][i].strip(".csv")

year=pd.DataFrame({'year':year})

trade=pd.DataFrame({'No.trades':trade})

win=pd.DataFrame({'Win%':win})

real_gain=pd.DataFrame({'Average gain($)':real_gain})

loss=pd.DataFrame({'Average loss($)':loss})

payoff=pd.DataFrame({'Payoff ratio':payoff})

factor=pd.DataFrame({'Profit factor':factor})

#7272
result =pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)


model_name = ['pred_logistic','pred_decision','pred_naive','pred_randomforest','pred_knn','pred_neural','pred_voting','pred_gbm','pred_xgb']

result.to_csv('C:/Users/user/Desktop/강화학습_sample_trading/instance_selection_result/test_result_{}_dqn_dnn_epoch_300.csv'.format(model_name[0]),encoding='euc-kr')



# %%
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:23:29 2022

@author: user
"""

# %%

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
    test_7219[i]['pred_xgb']=pred_xgb[i]

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
    test_7219[i]['pred_xgb']=test_7219[i]['pred_xgb'].astype('float')
    
    

#새로운 라벨 추가(e -> 인덱스 번호)
for i in range(0,72):
    test_7219[i]['position']=None
    
                       
#randomforest
for i in range(0,72):
    for e in test_7219[i].index:
        try:
            if test_7219[i]['pred_decision'][e]+test_7219[i]['pred_decision'][e+1]==0:
                test_7219[i]['position'][e+1]='no action'
            elif test_7219[i]['pred_decision'][e]+test_7219[i]['pred_decision'][e+1]==2:
                test_7219[i]['position'][e+1]='holding'
            elif test_7219[i]['pred_decision'][e] > test_7219[i]['pred_decision'][e+1]:
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


stock_name=pd.DataFrame({'stock_name':file_list})


for i in range(len(test_data)):
    stock_name['stock_name'][i] =stock_name['stock_name'][i].strip(".csv")

year=pd.DataFrame({'year':year})

trade=pd.DataFrame({'No.trades':trade})

win=pd.DataFrame({'Win%':win})

real_gain=pd.DataFrame({'Average gain($)':real_gain})

loss=pd.DataFrame({'Average loss($)':loss})

payoff=pd.DataFrame({'Payoff ratio':payoff})

factor=pd.DataFrame({'Profit factor':factor})

#7272
result =pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)


model_name = ['pred_logistic','pred_decision','pred_naive','pred_randomforest','pred_knn','pred_neural','pred_voting','pred_gbm','pred_xgb']

result.to_csv('C:/Users/user/Desktop/강화학습_sample_trading/instance_selection_result/test_result_{}_dqn_dnn_epoch_300.csv'.format(model_name[1]),encoding='euc-kr')


# %%
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:23:29 2022

@author: user
"""

# %%

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
    test_7219[i]['pred_xgb']=pred_xgb[i]

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
    test_7219[i]['pred_xgb']=test_7219[i]['pred_xgb'].astype('float')
    
    

#새로운 라벨 추가(e -> 인덱스 번호)
for i in range(0,72):
    test_7219[i]['position']=None
    
                       
#randomforest
for i in range(0,72):
    for e in test_7219[i].index:
        try:
            if test_7219[i]['pred_naive'][e]+test_7219[i]['pred_naive'][e+1]==0:
                test_7219[i]['position'][e+1]='no action'
            elif test_7219[i]['pred_naive'][e]+test_7219[i]['pred_naive'][e+1]==2:
                test_7219[i]['position'][e+1]='holding'
            elif test_7219[i]['pred_naive'][e] > test_7219[i]['pred_naive'][e+1]:
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


stock_name=pd.DataFrame({'stock_name':file_list})


for i in range(len(test_data)):
    stock_name['stock_name'][i] =stock_name['stock_name'][i].strip(".csv")

year=pd.DataFrame({'year':year})

trade=pd.DataFrame({'No.trades':trade})

win=pd.DataFrame({'Win%':win})

real_gain=pd.DataFrame({'Average gain($)':real_gain})

loss=pd.DataFrame({'Average loss($)':loss})

payoff=pd.DataFrame({'Payoff ratio':payoff})

factor=pd.DataFrame({'Profit factor':factor})

#7272
result =pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)


model_name = ['pred_logistic','pred_decision','pred_naive','pred_randomforest','pred_knn','pred_neural','pred_voting','pred_gbm','pred_xgb']

result.to_csv('C:/Users/user/Desktop/강화학습_sample_trading/instance_selection_result/test_result_{}_dqn_dnn_epoch_300.csv'.format(model_name[2]),encoding='euc-kr')

# %%
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:23:29 2022

@author: user
"""

# %%

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
    test_7219[i]['pred_xgb']=pred_xgb[i]

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
    test_7219[i]['pred_xgb']=test_7219[i]['pred_xgb'].astype('float')
    
    

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


stock_name=pd.DataFrame({'stock_name':file_list})


for i in range(len(test_data)):
    stock_name['stock_name'][i] =stock_name['stock_name'][i].strip(".csv")

year=pd.DataFrame({'year':year})

trade=pd.DataFrame({'No.trades':trade})

win=pd.DataFrame({'Win%':win})

real_gain=pd.DataFrame({'Average gain($)':real_gain})

loss=pd.DataFrame({'Average loss($)':loss})

payoff=pd.DataFrame({'Payoff ratio':payoff})

factor=pd.DataFrame({'Profit factor':factor})

#7272
result =pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)


model_name = ['pred_logistic','pred_decision','pred_naive','pred_randomforest','pred_knn','pred_neural','pred_voting','pred_gbm','pred_xgb']

result.to_csv('C:/Users/user/Desktop/강화학습_sample_trading/instance_selection_result/test_result_{}_dqn_dnn_epoch_300.csv'.format(model_name[3]),encoding='euc-kr')

# %%
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:23:29 2022

@author: user
"""

# %%

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
    test_7219[i]['pred_xgb']=pred_xgb[i]

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
    test_7219[i]['pred_xgb']=test_7219[i]['pred_xgb'].astype('float')
    
    

#새로운 라벨 추가(e -> 인덱스 번호)
for i in range(0,72):
    test_7219[i]['position']=None
    
                       
#randomforest
for i in range(0,72):
    for e in test_7219[i].index:
        try:
            if test_7219[i]['pred_knn'][e]+test_7219[i]['pred_knn'][e+1]==0:
                test_7219[i]['position'][e+1]='no action'
            elif test_7219[i]['pred_knn'][e]+test_7219[i]['pred_knn'][e+1]==2:
                test_7219[i]['position'][e+1]='holding'
            elif test_7219[i]['pred_knn'][e] > test_7219[i]['pred_knn'][e+1]:
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


stock_name=pd.DataFrame({'stock_name':file_list})


for i in range(len(test_data)):
    stock_name['stock_name'][i] =stock_name['stock_name'][i].strip(".csv")

year=pd.DataFrame({'year':year})

trade=pd.DataFrame({'No.trades':trade})

win=pd.DataFrame({'Win%':win})

real_gain=pd.DataFrame({'Average gain($)':real_gain})

loss=pd.DataFrame({'Average loss($)':loss})

payoff=pd.DataFrame({'Payoff ratio':payoff})

factor=pd.DataFrame({'Profit factor':factor})

#7272
result =pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)


model_name = ['pred_logistic','pred_decision','pred_naive','pred_randomforest','pred_knn','pred_neural','pred_voting','pred_gbm','pred_xgb']

result.to_csv('C:/Users/user/Desktop/강화학습_sample_trading/instance_selection_result/test_result_{}_dqn_dnn_epoch_300.csv'.format(model_name[4]),encoding='euc-kr')

# %%
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:23:29 2022

@author: user
"""

# %%

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
    test_7219[i]['pred_xgb']=pred_xgb[i]

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
    test_7219[i]['pred_xgb']=test_7219[i]['pred_xgb'].astype('float')
    
    

#새로운 라벨 추가(e -> 인덱스 번호)
for i in range(0,72):
    test_7219[i]['position']=None
    
                       
#randomforest
for i in range(0,72):
    for e in test_7219[i].index:
        try:
            if test_7219[i]['pred_neural'][e]+test_7219[i]['pred_neural'][e+1]==0:
                test_7219[i]['position'][e+1]='no action'
            elif test_7219[i]['pred_neural'][e]+test_7219[i]['pred_neural'][e+1]==2:
                test_7219[i]['position'][e+1]='holding'
            elif test_7219[i]['pred_neural'][e] > test_7219[i]['pred_neural'][e+1]:
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


stock_name=pd.DataFrame({'stock_name':file_list})


for i in range(len(test_data)):
    stock_name['stock_name'][i] =stock_name['stock_name'][i].strip(".csv")

year=pd.DataFrame({'year':year})

trade=pd.DataFrame({'No.trades':trade})

win=pd.DataFrame({'Win%':win})

real_gain=pd.DataFrame({'Average gain($)':real_gain})

loss=pd.DataFrame({'Average loss($)':loss})

payoff=pd.DataFrame({'Payoff ratio':payoff})

factor=pd.DataFrame({'Profit factor':factor})

#7272
result =pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)


model_name = ['pred_logistic','pred_decision','pred_naive','pred_randomforest','pred_knn','pred_neural','pred_voting','pred_gbm','pred_xgb']

result.to_csv('C:/Users/user/Desktop/강화학습_sample_trading/instance_selection_result/test_result_{}_dqn_dnn_epoch_300.csv'.format(model_name[5]),encoding='euc-kr')

# %%
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:23:29 2022

@author: user
"""

# %%

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
    test_7219[i]['pred_xgb']=pred_xgb[i]

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
    test_7219[i]['pred_xgb']=test_7219[i]['pred_xgb'].astype('float')
    
    

#새로운 라벨 추가(e -> 인덱스 번호)
for i in range(0,72):
    test_7219[i]['position']=None
    
                       
#randomforest
for i in range(0,72):
    for e in test_7219[i].index:
        try:
            if test_7219[i]['pred_voting'][e]+test_7219[i]['pred_voting'][e+1]==0:
                test_7219[i]['position'][e+1]='no action'
            elif test_7219[i]['pred_voting'][e]+test_7219[i]['pred_voting'][e+1]==2:
                test_7219[i]['position'][e+1]='holding'
            elif test_7219[i]['pred_voting'][e] > test_7219[i]['pred_voting'][e+1]:
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


stock_name=pd.DataFrame({'stock_name':file_list})


for i in range(len(test_data)):
    stock_name['stock_name'][i] =stock_name['stock_name'][i].strip(".csv")

year=pd.DataFrame({'year':year})

trade=pd.DataFrame({'No.trades':trade})

win=pd.DataFrame({'Win%':win})

real_gain=pd.DataFrame({'Average gain($)':real_gain})

loss=pd.DataFrame({'Average loss($)':loss})

payoff=pd.DataFrame({'Payoff ratio':payoff})

factor=pd.DataFrame({'Profit factor':factor})

#7272
result =pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)


model_name = ['pred_logistic','pred_decision','pred_naive','pred_randomforest','pred_knn','pred_neural','pred_voting','pred_gbm','pred_xgb']

result.to_csv('C:/Users/user/Desktop/강화학습_sample_trading/instance_selection_result/test_result_{}_dqn_dnn_epoch_300.csv'.format(model_name[6]),encoding='euc-kr')

# %%
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:23:29 2022

@author: user
"""

# %%

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
    test_7219[i]['pred_xgb']=pred_xgb[i]

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
    test_7219[i]['pred_xgb']=test_7219[i]['pred_xgb'].astype('float')
    
    

#새로운 라벨 추가(e -> 인덱스 번호)
for i in range(0,72):
    test_7219[i]['position']=None
    
                       
#randomforest
for i in range(0,72):
    for e in test_7219[i].index:
        try:
            if test_7219[i]['pred_gbm'][e]+test_7219[i]['pred_gbm'][e+1]==0:
                test_7219[i]['position'][e+1]='no action'
            elif test_7219[i]['pred_gbm'][e]+test_7219[i]['pred_gbm'][e+1]==2:
                test_7219[i]['position'][e+1]='holding'
            elif test_7219[i]['pred_gbm'][e] > test_7219[i]['pred_gbm'][e+1]:
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


stock_name=pd.DataFrame({'stock_name':file_list})


for i in range(len(test_data)):
    stock_name['stock_name'][i] =stock_name['stock_name'][i].strip(".csv")

year=pd.DataFrame({'year':year})

trade=pd.DataFrame({'No.trades':trade})

win=pd.DataFrame({'Win%':win})

real_gain=pd.DataFrame({'Average gain($)':real_gain})

loss=pd.DataFrame({'Average loss($)':loss})

payoff=pd.DataFrame({'Payoff ratio':payoff})

factor=pd.DataFrame({'Profit factor':factor})

#7272
result =pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)


model_name = ['pred_logistic','pred_decision','pred_naive','pred_randomforest','pred_knn','pred_neural','pred_voting','pred_gbm','pred_xgb']

result.to_csv('C:/Users/user/Desktop/강화학습_sample_trading/instance_selection_result/test_result_{}_dqn_dnn_epoch_300.csv'.format(model_name[7]),encoding='euc-kr')

# %%
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:23:29 2022

@author: user
"""

# %%

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
    test_7219[i]['pred_xgb']=pred_xgb[i]

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
    test_7219[i]['pred_xgb']=test_7219[i]['pred_xgb'].astype('float')
    
    

#새로운 라벨 추가(e -> 인덱스 번호)
for i in range(0,72):
    test_7219[i]['position']=None
    
                       
#randomforest
for i in range(0,72):
    for e in test_7219[i].index:
        try:
            if test_7219[i]['pred_xgb'][e]+test_7219[i]['pred_xgb'][e+1]==0:
                test_7219[i]['position'][e+1]='no action'
            elif test_7219[i]['pred_xgb'][e]+test_7219[i]['pred_xgb'][e+1]==2:
                test_7219[i]['position'][e+1]='holding'
            elif test_7219[i]['pred_xgb'][e] > test_7219[i]['pred_xgb'][e+1]:
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


stock_name=pd.DataFrame({'stock_name':file_list})


for i in range(len(test_data)):
    stock_name['stock_name'][i] =stock_name['stock_name'][i].strip(".csv")

year=pd.DataFrame({'year':year})

trade=pd.DataFrame({'No.trades':trade})

win=pd.DataFrame({'Win%':win})

real_gain=pd.DataFrame({'Average gain($)':real_gain})

loss=pd.DataFrame({'Average loss($)':loss})

payoff=pd.DataFrame({'Payoff ratio':payoff})

factor=pd.DataFrame({'Profit factor':factor})

#7272
result =pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)


model_name = ['pred_logistic','pred_decision','pred_naive','pred_randomforest','pred_knn','pred_neural','pred_voting','pred_gbm','pred_xgb']

result.to_csv('C:/Users/user/Desktop/강화학습_sample_trading/instance_selection_result/test_result_{}_dqn_dnn_epoch_300.csv'.format(model_name[8]),encoding='euc-kr')