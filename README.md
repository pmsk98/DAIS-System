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

### 2. Instance Selection 
- Run Instance Selection with Machine Learning.py

### 3. Evaluation 
- Run Instance Selection Result.py
  
## Study result
### DAIS System Trading signal 
![DAIS Trading singal_box 추가](https://github.com/pmsk98/DAIS-System/assets/45275607/c0bd0487-b12f-412e-b3ab-5a94f6a7d88f)
![instance_selection_payoff_ratio_boxplot](https://github.com/pmsk98/DAIS-System/assets/45275607/7b19aebf-3293-4fdd-8942-3a771cb6be85)
![instance_selection_profit_factor_boxplot](https://github.com/pmsk98/DAIS-System/assets/45275607/0f367ab8-8f8f-4925-a206-ed5cc76c0b01)
