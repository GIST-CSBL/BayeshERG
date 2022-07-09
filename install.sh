#!/bin/bash
echo Input CUDA version of your GPU, ex. 10.2
read cuda_ver
echo DGL and Pytorch with CUDA v$cuda_ver will be installed. 
conda install -c dglteam dgl-cuda$cuda_ver=0.4.3 -y
conda install pytorch=1.10.1 cudatoolkit=$cuda_ver -c pytorch -y
conda install requests -y
conda install tqdm -y
conda install -c conda-forge mdtraj -y
