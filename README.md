# SHGNN
The source code and dataset of the paper: [SHGNN: Structure-Aware Heterogeneous Graph Neural Network](https://arxiv.org/abs/2112.06244).
![image](https://user-images.githubusercontent.com/25242325/145964701-e52f7934-06b9-423a-b9af-e114693015e4.png)

## Requirements
The framework is implemented using python3 with dependencies specified in [requirements.txt](https://github.com/Wentao-Xu/SHGNN/blob/main/requirements.txt).
```
git clone https://github.com/Wentao-Xu/SHGNN.git
cd SHGNN
conda create -n shgnn python=3.8
conda activate shgnn
pip install -r requirements.txt
```

## Dataset preparation
```
source prepare_data.sh
tar -zxvf data.tar.gz
mkdir checkpoint
```
## Running the code
```
# IMDB dataset
python run_IMDB.py

# DBLP dataset
python run_DBLP.py

# ACM dataset
python run_ACM.py
```
