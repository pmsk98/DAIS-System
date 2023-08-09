# A Novel Trading System for the Stock Market using Deep Q Network Action and Instance Selection


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
- Using [SAHI](https://github.com/obss/sahi) (Slicing Aided Hyper Inference)
- But... very bad performance :(
- Need to explore other methods
## Study result
### Glomerulus segmentation (HuBMAP Dataset)
![image](https://github.com/SCH-YcHan/Glomer/assets/113504815/14bd08fd-62c7-4097-a3d6-130d00584bf2)
![image](https://github.com/SCH-YcHan/Glomer/assets/113504815/138dc0df-81f9-4515-8b53-00c4fd4a8c8f)
