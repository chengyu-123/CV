import cv2


def illuminance_compensation(image):
    # 將圖像轉換為LAB色彩空間
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 分離LAB圖像的L、A和B通道
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # 創建CLAHE（對比度有限自適應直方圖均衡化）對象，設置參數
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # 對L通道應用CLAHE
    clahe_l_channel = clahe.apply(l_channel)
    
    # 將CLAHE增強的L通道與原始A和B通道合併
    enhanced_lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))
    
    # 將增強的LAB圖像轉換回BGR色彩空間
    enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)
    
    return enhanced_image