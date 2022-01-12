#!/bin/bash

conda install -c dglteam dgl-cuda10.1=0.4.3 -y
conda install pytorch cudatoolkit=10.1 -c pytorch -y
conda install requests -y
conda install tqdm -y
conda install -c conda-forge mdtraj -y
