#!/bin/bash

# 创建 Conda 环境
conda create -n tie python==3.8

# 激活 Conda 环境
conda activate tie

# 安装软件包
pip install numpy
pip install gym==0.23
pip install wandb
pip install matplotlib
pip install torch_geometric

# 安装 PyTorch（仅适用于 Linux 系统）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
