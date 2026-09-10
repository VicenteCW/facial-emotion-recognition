#!/bin/bash

# 樹莓派自動安裝腳本 / Raspberry Pi Auto-Installation Script
# 用於自動安裝所有依賴和配置 / Automatically installs all dependencies

set -e  # Exit on error

echo "=========================================="
echo "人臉情感識別系統 - 樹莓派自動安裝"
echo "Facial Emotion Recognition - Raspberry Pi Setup"
echo "=========================================="
echo ""

# 檢查 Python 版本 / Check Python version
echo "[1/5] Checking Python version..."
python3 --version
echo ""

# 更新系統 / Update system
echo "[2/5] Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y
echo "✓ System updated"
echo ""

# 安裝依賴 / Install dependencies
echo "[3/5] Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libatlas-base-dev \
    libjasper-dev \
    libtiff5 \
    libjasper1 \
    libharfbuzz0b \
    libwebp6 \
    libopenjp2-7 \
    libopenjp2-7-dev \
    libtiff-dev \
    python3-venv \
    git

echo "✓ System dependencies installed"
echo ""

# 創建虛擬環境 / Create virtual environment
echo "[4/5] Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# 激活虛擬環境 / Activate virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# 安裝 Python 依賴 / Install Python dependencies
echo "[5/5] Installing Python dependencies..."
pip3 install --upgrade pip setuptools wheel

# 嘗試從 piwheels 安裝 TensorFlow (更快) / Try installing from piwheels
echo "Installing TensorFlow (this may take a while on Raspberry Pi)..."
pip3 install --index-url https://www.piwheels.org/simple tensorflow || \
    pip3 install tensorflow

# 安裝其他依賴 / Install other dependencies
pip3 install -r requirements.txt

echo "✓ Python dependencies installed"
echo ""

# 增加交換空間 / Increase swap space
echo "[Optional] Increasing swap space for better performance..."
read -p "Do you want to increase swap size to 2GB? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo dphys-swapfile swapoff
    sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
    sudo dphys-swapfile swapon
    echo "✓ Swap space increased to 2GB"
fi
echo ""

# 完成 / Completion
echo "=========================================="
echo "✓ 安裝完成! / Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Test with camera:"
echo "   python3 emotion_recognition.py"
echo ""
echo "3. Or test with a single image:"
echo "   python3 simple_demo.py /path/to/image.jpg"
echo ""
echo "For more information, see README.md"
echo ""
