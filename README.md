# 人臉情感識別系統 - Raspberry Pi 版本

A Python-based facial emotion recognition system optimized for Raspberry Pi using deep learning.

## 功能特性

✅ **人臉檢測** - 使用 Haar Cascade 進行實時人臉檢測  
✅ **情感識別** - 識別 7 種情感（生氣、厭惡、害怕、開心、中立、悲傷、驚訝）  
✅ **樹莓派優化** - 輕量級模型，適合低功耗設備  
✅ **實時處理** - 支持攝像頭實時視頻流處理  
✅ **模型訓練** - 支持 FER2013 數據集或自定義數據集  

## 系統要求

### 硬件要求
- Raspberry Pi 4 (推薦) 或 Raspberry Pi 3B+ 及以上
- 至少 2GB RAM (4GB 推薦)
- USB 攝像頭 或 Raspberry Pi 官方相機模塊
- 20GB+ SD 卡

### 軟件要求
- Python 3.7 或更高版本
- Raspberry Pi OS (基於 Debian)

## 安裝指南

### 1. 準備 Raspberry Pi

```bash
# 更新系統
sudo apt-get update
sudo apt-get upgrade

# 安裝必要的依賴
sudo apt-get install python3-pip
sudo apt-get install libatlas-base-dev
sudo apt-get install libjasper-dev
sudo apt-get install libharfbuzz0b
sudo apt-get install libwebp6
sudo apt-get install libtiff5
sudo apt-get install libjasper1
sudo apt-get install libopenjp2-7
sudo apt-get install libopenjp2-7-dev
sudo apt-get install libtiff-dev
sudo apt-get install libcamera-dev
```

### 2. 克隆倉庫

```bash
git clone https://github.com/VicenteCW/facial-emotion-recognition.git
cd facial-emotion-recognition
```

### 3. 創建虛擬環境 (推薦)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. 安裝依賴

```bash
# 方案 A: 使用 pip 安裝 (適合新手)
pip3 install -r requirements.txt

# 方案 B: 為樹莓派優化的安裝
pip3 install opencv-python
pip3 install numpy==1.24.3
pip3 install --index-url https://www.piwheels.org/simple tensorflow
pip3 install scikit-learn
```

### 5. 安裝樹莓派專用加速庫 (可選但推薦)

```bash
# 安裝 OpenCV 樹莓派優化版本
pip3 install opencv-contrib-python

# 安裝 TensorFlow Lite (更輕量)
pip3 install tensorflow-lite
```

## 快速開始

### 1. 運行實時情感識別

```bash
python3 emotion_recognition.py
```

按 'q' 鍵退出程序。

### 2. 使用預訓練模型

```python
from emotion_recognition import RaspberryPiEmotionApp

# 使用已訓練的模型
app = RaspberryPiEmotionApp(model_path='emotion_model.h5')
app.run()
```

### 3. 訓練自定義模型

#### 準備數據集

```
dataset/
├── angry/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

#### 運行訓練

```bash
# 編輯 train_model.py 中的 main() 函數，設置正確的數據集路徑
python3 train_model.py
```

## 代碼示例

### 基本使用

```python
from emotion_recognition import RaspberryPiEmotionApp

# 創建應用實例
app = RaspberryPiEmotionApp()

# 運行應用
app.run()
```

### 自定義使用

```python
from emotion_recognition import EmotionRecognizer
import cv2

# 初始化識別器
recognizer = EmotionRecognizer(model_path='emotion_model.h5')

# 讀取圖像
img = cv2.imread('face.jpg')

# 檢測並識別情感
output_frame, emotions = recognizer.detect_and_recognize(img)

# 顯示結果
cv2.imshow('Result', output_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 打印結果
for emotion_data in emotions:
    print(f"情感: {emotion_data['emotion']}, 置信度: {emotion_data['confidence']:.2f}")
```

### 單張圖像情感識別

```python
from emotion_recognition import EmotionRecognizer
import cv2

recognizer = EmotionRecognizer()

# 讀取人臉圖像
face = cv2.imread('single_face.jpg')

# 預測情感
emotion, confidence = recognizer.predict_emotion(face)

print(f"檢測到的情感: {emotion}")
print(f"置信度: {confidence:.4f}")
```

## 性能優化建議

### 1. 降低分辨率

```python
# 在 emotion_recognition.py 的 start_camera() 方法中修改:
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)   # 從 320 降至 240
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)  # 從 240 降至 180
self.cap.set(cv2.CAP_PROP_FPS, 10)            # 從 15 降至 10 FPS
```

### 2. 使用多線程處理

```python
import threading

def process_frames():
    # 在後臺線程進行處理
    while app.is_running:
        # 檢測和識別
        frame_output, emotions = recognizer.detect_and_recognize(frame)

# 創建線程
thread = threading.Thread(target=process_frames)
thread.start()
```

### 3. 啟用硬件加速

```bash
# 安裝 OpenCV 硬件加速支持
sudo apt-get install libopencv-dev
```

## 數據集

### FER2013 數據集

下載地址: [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

包含 35,887 張 48×48 像素的灰度人臉圖像，分為 7 個情感類別。

### 自定義數據集

確保以下目錄結構:

```
dataset/
├── angry/        # 生氣
├── disgust/      # 厭惡
├── fear/         # 害怕
├── happy/        # 開心
├── neutral/      # 中立
├── sad/          # 悲傷
└── surprise/     # 驚訝
```

## 故障排除

### 1. 無法打開攝像頭

```bash
# 檢查攝像頭是否連接
ls /dev/video*

# 檢查攝像頭權限
sudo usermod -a -G video $USER

# 重新啟動 Raspberry Pi
sudo reboot
```

### 2. 內存不足

```bash
# 增加交換空間
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# 將 CONF_SWAPSIZE=100 改為 CONF_SWAPSIZE=2048
sudo dphys-swapfile on
```

### 3. 速度緩慢

- 降低分辨率和幀率
- 減少 FPS 設置
- 使用 TensorFlow Lite 版本
- 優化模型大小

## 項目結構

```
facial-emotion-recognition/
├── emotion_recognition.py    # 主程序
├── train_model.py            # 模型訓練
├── simple_demo.py            # 簡單示例
├── requirements.txt          # 依賴列表
├── raspberry_pi_setup.sh    # 自動安裝腳本
└── README.md                 # 本文檔
```

## 架構說明

### 系統流程

```
視頻流 → 人臉檢測 → 人臉預處理 → 深度學習模型 → 情感分類 → 結果顯示
```

### 模型架構

```
輸入: 48×48 灰度圖像
  ↓
Conv2D(32, 3×3) + ReLU
  ↓
MaxPooling2D(2×2) + Dropout(0.25)
  ↓
Conv2D(64, 3×3) + ReLU
  ↓
MaxPooling2D(2×2) + Dropout(0.25)
  ↓
Conv2D(128, 3×3) + ReLU
  ↓
MaxPooling2D(2×2) + Dropout(0.25)
  ↓
Flatten
  ↓
Dense(256) + ReLU + Dropout(0.5)
  ↓
Dense(7, Softmax)
  ↓
輸出: 7 種情感概率
```

## 許可證

MIT License - 詳見 LICENSE 文件

## 貢獻

歡迎提交 Issue 和 Pull Request！

## 聯繫方式

如有問題，請在 GitHub 上提交 Issue。

## 參考資源

- [TensorFlow 官方文檔](https://www.tensorflow.org/)
- [OpenCV 官方文檔](https://docs.opencv.org/)
- [Raspberry Pi 官方文檔](https://www.raspberrypi.org/documentation/)
- [FER2013 論文](https://arxiv.org/abs/1307.0414)

---

**最後更新**: 2026-09-10  
**維護者**: VicenteCW
