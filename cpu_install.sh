#!/bin/bash
conda install pytorch=1.10.1 cpuonly -c pytorch -y
conda install -c dglteam dgl=0.4.3 -y
conda install requests -y
conda install tqdm -y
conda install -c conda-forge mdtraj -y
