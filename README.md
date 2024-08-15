# DACIN: Deep Adversarial Clustering-driven Imputation Networks

![Python 3.11](https://img.shields.io/badge/Python-3.11-green)
![MIT License](https://img.shields.io/badge/License-MIT-blue)

## Requirements

Execute the following command to install the required libraries:
```
pip install -r requirements.txt
```

## Data

The data used in this paper can be downloaded from:
-   UCI Letter (https://archive.ics.uci.edu/dataset/59/letter+recognitionn)
-   UCI Pendigits (https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits)
-   UCI Optdigits (https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)

## Example command

```
python main.py -data_dir dataset/letter/missing_ratio/ -batch_size 1474 -k 6 -epochs 1000 -iter_num 5
```
