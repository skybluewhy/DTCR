# DTCR
DTCR is a novel TCR generation model based on discrete diffusion. This model leverages corruption scheme to simulate the TCR mutation process and integrates advanced TCR-epitope binding prediction model to simulate the affinity screening process of TCRs,
providing a flexible and controllable framework for precise TCR generation. DTCR captures the discrete nature of TCR sequences and demonstrates superior generation performance in terms of specificity, biological conservation, and sequence diversity.

# Dependencies
DTCR is writen in Python based on Pytorch. The required software dependencies are listed below:
```
torch
torchvision
pytorch_lightning
fastNLP
lightning_utilities
torchmetrics
Bio
scikit-learn
transformers==4.21.1
numpy==1.23.4
```

# Installation
Setup conda environment:
```
# Create environment
conda create -n DTCR python=3.8 -y
conda activate DTCR

# Instaill requirements
conda install pytorch==1.8.1 torchvision==0.9.1 -c pytorch -y
pip install pytorch_lightning  --no-deps
pip install fastNLP
pip install lightning_utilities
pip install torchmetrics
pip install Bio
pip install scikit-learn
pip install transformers==4.21.1
pip install numpy==1.23.4

# Clone DTCR
git clone https://github.com/skybluewhy/DTCR.git
cd DTCR
```

# Data
All the data used in the paper were collected from public databases: VDJdb, McAPS-TCR, IEDB and MIRA.

# Usage of MFTEP
Data Preparation:
Prepare the train dataset () in <BASE_FOLDER>/data/.


Training DTCR with a BLOSUM matrix as transition matrix:
```
python main_conditional_epitope_design.py --schedule_name blosum --schedule exp --batch_size 128 --epochs 20
```
Generate novel TCRs using BLOSUM transition matrix based DTCR:
```
python TCR_design_cond_epitope.py --seed_num 2000 --batch_size 1024 --model_path "./checkpoint/model_design19.th" --schedule_name 'blosum' --schedule exp
```

Training DTCR with a Random-base matrix as transition matrix:
```
python main_conditional_epitope_design.py --schedule_name random --schedule mutual --batch_size 128 --epochs 20
```
Generate novel TCRs using Random transition matrix based DTCR:
```
python TCR_design_cond_epitope.py --seed_num 2000 --batch_size 1024 --model_path "./checkpoint/model_design19.th" --schedule_name 'random' --schedule mutual
```

