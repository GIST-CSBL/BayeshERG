# BayeshERG : A Bayesian Graph Neural Network for predicting hERG blockers
BayeshERG Official Repository

# Prerequsites
- Anaconda

To avoid the package version issue, we open our code with Anaconda virtual environment. Therefore, the Anaconda should be installed in advance.

# Requirements
### Input Format 

Any `.csv` file with `smiles` column.

(Example)
|  ID  |  smiles |
| -----|---------|

# Usage
### Create conda virtual environment

```
$conda create -n env BayeshERG -c conda-forge rdkit python=3.6
```
### Activate the virtual environment
```
$conda activate BayeshERG
```

### Install dependencies
- If your system has GPU, check the CUDA version in advance (nvidia-smi).
Excute the installation shell script    
    
```
$sh install.sh
```
Then, type the cuda version to the shell and press enter.
  
- If your system has no GPU, excute the cpu version dependencies.
```
$sh cpu_install.sh
```

### Prediction
```
usage: $python main.py  [-i] input_csv_file_path 
                        [-o] output_file_name 
                        [-c] 'cpu' or 'gpu' (default 'cpu')
                        [-t] sampling time (integer, default 30)
```
- Example

_With GPU_
```
$python main.py -i data/External/EX1.csv -o EX1_pred -c gpu -t 30
```
_With CPU_
```
$python main.py -i data/External/EX1.csv -o EX1_pred -c cpu -t 30
```
### Output Format

The prediction results (Prediction score, Uncertainties) are appened to the input `.csv` file and saved to `prediction_results` directory as `output_file_name.csv`.
|  ID  |  smiles  |  score  |  alea  |  epis  |
| -----|----------|---------|--------|--------|

Also, the attention images(`.svg`) are also depicted and saved to `attention_results/output_file_name` directory.

# Contact
hyunhokim@gm.gist.ac.kr

15pms@gist.ac.kr

hjnam@gist.ac.kr
