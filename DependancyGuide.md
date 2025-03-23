# Dependancy guide

## Dependancy installation guide
Recreation of environment used to run this repository on a new computer; 
Install Anaconda
'''
source ~/anaconda3/bin/activate
conda --version
conda env create -f environment.yml
conda activate cpp_env
conda activate cpp_env
mkdir build && cd build
cmake ..
make
'''

## Usage of conda environment
Do this when new terminal is used;
'''
source ~/anaconda3/bin/activate thrust_alloc_cpp_env
'''

When done with terminal, close it or do
'''
conda deactivate
'''

### Dependancy update guide
Exporting environment to a YAML file; 
'''
conda env export > environment.yml
'''


