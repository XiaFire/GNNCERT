# GNNCert: Deterministic Certification of Graph Neural Networks against Adversarial Perturbations

An official PyTorch implementation of "[GNNCert: Deterministic Certification of Graph Neural Networks against Adversarial Perturbations](https://openreview.net/forum?id=IGzaH538fz)" (ICLR 2024).

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

To run GNNCert with GIN, use the following commands:

+ For feature division:

```bash
bash jobs/feature_division.sh
bash jobs/feature_division-test.sh
```

- For structure division:

```bash
bash jobs/structure_division.sh
bash jobs/structure_division-test.sh
```

- For feature-structure jointly division:

```bash
bash jobs/method_structure_feature.sh
bash jobs/method_structure_feature-test.sh
```

You can also try different architectures, such as GCN and GAT:

- For GCN:

```bash
bash jobs/GCN.sh
bash jobs/GCN-test.sh
```

- For GAT:

```bash
bash jobs/GAT.sh
bash jobs/GAT-test.sh
```
