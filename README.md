# Optimal Constraint Thruster Allocation README


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

Adding or updating environment.yml after chenging dependancies
'''
git add environment.yml
git commit -m "Dependacy update in environment.yml file"
git pull
git push
'''

## Questions

- Forslag som er bedre enn PID-en jeg bruker som motion_controller nå?
- Er ran() modellen riktig implementert, se spessielt på forskjellene fra otter.m (azimuth fuksjonaliteten)
- Sliter med endring av heading, har satt opp tidskonstanten for yaw i ran modellen og begrenset U_max til 0.1 for å unngå problematikk forårsaket av treg endring i heading. Jeg veit båten på ekte kan snu seg på stedet veldig fort, så det er ikke urealistisk å implementere.
- Er guidance ok? Har implementert veldig enkel guidance, men fungerer DP og stationkeeping        prinsippene over avtander på 20 m? 

- Mange parametere er en gjetning, har ikke alle ekte tall (Som tall fra Bollard) ennå. 