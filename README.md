# DoAnNghienCuu

## Setup Environment

It is recommended to use `conda` or `venv`.

### Using `conda` (recommended)

```bash
conda create -n DANC python=3.9 -y
conda activate DANC
pip install -r requirements.txt
```

### Using `venv`
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
cd Code
pip install -r requirements.txt
```
## Run
#### MultiMNIST
```bash
#training
python mdmtn_mm_mgda.py
python mdmtn_mm.py
#infer
python infer_mm_mgda.py
python infer_mm.py
# Plot 2D Pareto
python twoDpf_study_mdmtn_mm.py
```
#### Cifar10Mnist
```bash
#training
python mdmtn_cm_mgda.py
python mdmtn_cm.py
#infer
python infer_cm_mgda.py
python infer_cm.py
# Plot 2D Pareto
python twoDpf_study_mdmtn_cm.py
```