# BayeshERG : A Bayesian Graph Neural Network for predicting hERG blockers
This repository is BayeshERG official repository. It contains the pytorch implementation of BayeshERG and trained model to predict arbitrary compounds. 
The implementation of BayehERG has referred to the official implementation of related studies [1-3].   

The BayeshERG is developed with the python v3.6 and following packages:`dgl`, `pytorch`, and `rdkit`.


# License

BayeshERG follows [GPL 3.0v license](LICENSE). Therefore, BayeshERG is open source and free to use for everyone.

However, compounds which are found by using BayeshERG follows [CC-BY-NC-4.0](CC-BY-NC-SA-4.0). Thus, those compounds are freely available for academic purposes or individual research but restricted for commercial use.

# Contact

hjnam@gist.ac.kr

hyunhokim@gm.gist.ac.kr

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
### Create conda virtual environment and install dependencies
Conda environment file 'environment.yml' is provided
```
$ conda env create --name BayeshERG --file=environment.yml
```
### Activate the virtual environment
```
$ conda activate BayeshERG
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
Hyunho Kim, hyunhokim@gm.gist.ac.kr

Minsu Park, 15pms@gist.ac.kr

Hojung Nam (Corresponding Author), hjnam@gist.ac.kr

# Reference
[1] Gal, Yarin, Jiri Hron, and Alex Kendall. "Concrete dropout." arXiv preprint arXiv:1705.07832 (2017).

[2] Scalia, Gabriele, et al. "Evaluating scalable uncertainty estimation methods for deep learning-based molecular property prediction." Journal of chemical information and modeling 60.6 (2020): 2697-2717.

[3] Yang, Kevin, et al. "Analyzing learned molecular representations for property prediction." Journal of chemical information and modeling 59.8 (2019): 3370-3388.


---
@ Last modified : 2022.11.18
