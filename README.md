# GraphGuard: Provably Robust Graph Classification against Adversarial Attacks

An official PyTorch implementation of "[GraphGuard: Provably Robust Graph Classification against Adversarial Attacks](https://openreview.net/forum?id=IGzaH538fz)" (ICLR 2024).

This code is based on the official GIN implementation: https://github.com/weihua916/powerful-gnns.

## Installation

First, unzip the dataset.zip file.

```bash
cd GraphGuard
unzip dataset.zip
```

Then, create a new environment and install dependencies. We use Python 3.8 and PyTorch 1.10.0+cu113.

```bash
pip install -r requirements.txt
```

*Note:* In addition to the datasets in dataset.zip, you can also use the datasets from PyG. To use the PyG datasets, install the corresponding version of PyG according to https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html.

## Experments

+ To run GraphGuard with GIN, using command:

```
# For feauture division
bash jobs/feature_division.sh
bash jobs/feature_division-test.sh

# For structure division
bash jobs/structure_division.sh
bash jobs/structure_division-test.sh

# For feauture-structure jointly division
bash jobs/method_structure_feature.sh
bash jobs/method_structure_feature-test.sh
```

+ You can also try different architecture.

```
# For GCN
bash GCN.sh
bash GCN-test.sh

# For GAT
bash GAT.sh
bash GAT-test.sh
```
