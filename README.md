# BayeshERG : A Bayesian Graph Neural Network for predicting hERG blockers
This repository is BayeshERG official repository. It contains the pytorch implementation of BayeshERG and trained model to predict arbitrary compounds. 
The implementation of BayehERG has referred to the official implementation of related studies [1-3].   

The BayeshERG is developed with the python v3.6 and following packages:`dgl`, `pytorch`, and `rdkit`.




# Prerequsites
- Anaconda

To avoid the package version issue, we open our code with Anaconda virtual environment. Therefore, the Anaconda should be installed in advance.
https://www.anaconda.com/products/individual

# Requirements
### Input Format 

Any `.csv` file with `smiles` column.

(Example)
|  ID  |  smiles |
| -----|---------|

# Usage
### Create conda virtual environment

```
$ conda create -n env BayeshERG -c conda-forge rdkit python=3.6
```
### Activate the virtual environment
```
$ conda activate BayeshERG
```

### Install dependencies
- If your system has GPU, check the CUDA version in advance (nvidia-smi). 

Excute the installation shell script `install.sh`    
    
```
$ sh install.sh
```
Then, type the cuda version to the shell and press enter.

```
$ sh install.sh
Input CUDA version of your GPU, ex. 10.2
: 10.2 (Enter)
DGL and Pytorch with CUDA v10.2 will be installed.
...
...
```
  
- If your system has no GPU, excute the cpu-version shell script `cpu_install.sh`.
```
$ sh cpu_install.sh
```

### Prediction
```
usage: $ python main.py [-i] input_csv_file_path 
                        [-o] output_file_name 
                        [-c] 'cpu' or 'gpu' (default 'cpu')
                        [-t] sampling time (integer, default 30)
```
- Example

```
// With GPU
$ python main.py -i data/External/EX1.csv -o EX1_pred -c gpu -t 30

// With CPU
$ python main.py -i data/External/EX1.csv -o EX1_pred -c cpu -t 30
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

# Reference
[1] Gal, Yarin, Jiri Hron, and Alex Kendall. "Concrete dropout." arXiv preprint arXiv:1705.07832 (2017).

[2] Scalia, Gabriele, et al. "Evaluating scalable uncertainty estimation methods for deep learning-based molecular property prediction." Journal of chemical information and modeling 60.6 (2020): 2697-2717.

[3] Yang, Kevin, et al. "Analyzing learned molecular representations for property prediction." Journal of chemical information and modeling 59.8 (2019): 3370-3388.


---
@ Last modified : 2022.01.12
