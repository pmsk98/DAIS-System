# A Novel Trading System for the Stock Market using Deep Q-Network Action and Instance Selection
## Env setting (Local)
```
OS: Windows 10 Pro
CPU: Intel(R) Xeon(R) W-2225 CPU
GPU: NVIDIA GeForce RTX 3060 Ti
CUDA version: 11.7
CuDNN version: 8.4.0
Workstation: Anaconda3
```

## Order of execution
### 1. Reinforcement Learning 
#### Example.
```
python main.py --mode train --ver v3 --name 005930 --stock_code 005930 --rl_method a2c --net dnn --start_date 20180101 --end_date 20191231
```
##### DQN Reward Convergence
![DQN_Reward figure](https://github.com/pmsk98/DAIS-System/assets/45275607/07cce8b3-3917-4b60-92e3-d064d864ceda)


### 2. Instance Selection 
- Run Instance Selection with Machine Learning.py

### 3. Evaluation 
- Run Instance Selection Result.py
  
## Study result
### DAIS System Trading signal 
![DAIS Trading singal_box 추가](https://github.com/pmsk98/DAIS-System/assets/45275607/27ec4804-d80b-40ed-babe-1d951ffbc2bd)

### DAIS System Boxplot (Payoff ratio & Profit Factor)
#### Payoff ratio
![instance_selection_payoff_ratio_boxplot](https://github.com/pmsk98/DAIS-System/assets/45275607/7ac30e3a-d02d-4e6d-a5d0-9258e58d34ee)
#### Profit Factor
![instance_selection_profit_factor_boxplot](https://github.com/pmsk98/DAIS-System/assets/45275607/b9807162-0836-4ee6-93c4-9ea21b8c7764)


