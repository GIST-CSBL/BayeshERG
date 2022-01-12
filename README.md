# BayeshERG : A Bayesian Graph Neural Network for predicting hERG blockers
- BayeshERG Official Repository

# Prerequsites
- Anaconda

To avoid the package version issue, we open our code with Anaconda virtual environment. Therefore, the Anaconda should be installed in advance.

# Requirements
### Input Format 

Any `.csv` file with `smiles` column.

(Example)
|  ID  |  smiles |
| -----|---------|


### Output Format

The prediction results (Prediction score, Uncertainties) are appened to the input `.csv` file.
|  ID  |  smiles  |  score  |  alea  |  epis  |
| -----|----------|---------|--------|--------|



