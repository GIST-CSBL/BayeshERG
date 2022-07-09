#!/bin/bash

conda install -c dglteam dgl-cuda10.2=0.4.3 -y
conda install pytorch=1.10.1 cudatoolkit=10.2 -c pytorch -y
conda install requests -y
conda install tqdm -y
conda install -c conda-forge mdtraj -y
