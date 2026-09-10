"""
簡單示例 - 單張圖像情感識別
Simple Demo - Single Image Emotion Recognition
"""

import cv2
from emotion_recognition import EmotionRecognizer


def demo_single_image(image_path):
    """
    識別單張圖像中的人臉情感
    Recognize emotions in a single image
    
    Args:
        image_path: 圖像文件路徑 / Path to image file
    """
    print(f"Loading image from {image_path}...")
    
    # 創建識別器 / Create recognizer
    recognizer = EmotionRecognizer()
    
    # 讀取圖像 / Read image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # 檢測並識別情感 / Detect and recognize emotions
    output_image, emotions = recognizer.detect_and_recognize(image)
    
    # 打印結果 / Print results
    print(f"\n{'='*50}")
    print("Detection Results:")
    print(f"{'='*50}")
    
    if emotions:
        for i, emotion_data in enumerate(emotions, 1):
            print(f"\nFace #{i}:")
            print(f"  Emotion: {emotion_data['emotion']}")
            print(f"  Confidence: {emotion_data['confidence']:.4f}")
            print(f"  Bounding Box: {emotion_data['bbox']}")
    else:
        print("No faces detected in the image!")
    
    print(f"{'='*50}\n")
    
    # 顯示結果 / Display result
    cv2.imshow('Emotion Recognition Result', output_image)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 可選: 保存結果圖像 / Optional: Save result image
    output_path = image_path.replace('.jpg', '_result.jpg').replace('.png', '_result.png')
    cv2.imwrite(output_path, output_image)
    print(f"Result saved to {output_path}")


def demo_camera():
    """
    使用攝像頭進行實時情感識別
    Real-time emotion recognition using camera
    """
    print("Starting camera...")
    
    # 創建應用 / Create app
    from emotion_recognition import RaspberryPiEmotionApp
    app = RaspberryPiEmotionApp(camera_index=0)
    
    # 運行應用 / Run app
    app.run()


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("人臉情感識別系統 - 簡單示例")
    print("Facial Emotion Recognition - Simple Demo")
    print("="*60)
    print()
    
    # 檢查命令行參數 / Check command line arguments
    if len(sys.argv) > 1:
        # 如果提供了圖像路徑，使用該圖像 / If image path provided, use it
        demo_single_image(sys.argv[1])
    else:
        # 否則使用攝像頭 / Otherwise use camera
        print("Usage:")
        print("  1. Single image:  python3 simple_demo.py <image_path>")
        print("  2. Camera demo:   python3 simple_demo.py --camera")
        print()
        
        if len(sys.argv) > 1 and sys.argv[1] == '--camera':
            demo_camera()
        else:
            # 默認使用攝像頭 / Default to camera
            demo_camera()
