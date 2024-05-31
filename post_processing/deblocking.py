import cv2
import numpy as np

def deblockM(image, block_size=8):

    """
    使用均值濾波器進行去塊處理。

    參數:
    - image: 輸入圖像（numpy數組）。
    - block_size: 要平滑的塊的大小。預設為8。

    返回:
    - deblocked_image: 去塊後的圖像。
    """

    # 獲取圖像尺寸
    height, width = image.shape[:2]

    # 遍歷每個塊
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # 定義塊的邊界
            block_start_x = x
            block_end_x = min(x + block_size, width)
            block_start_y = y
            block_end_y = min(y + block_size, height)

            # 提取塊
            block = image[block_start_y:block_end_y, block_start_x:block_end_x]

            # 對塊應用均值濾波器
            mean_value = cv2.mean(block)[0]
            block[:, :] = mean_value

    return image

def deblockNLM(image_path, h=10, search_window=21, block_size=7):

    """
    使用非局部均值濾波器進行去塊處理。

    參數：
    - image_path：圖像文件的路徑。
    - h：濾波器強度參數，控制濾波的程度，默認為10。
    - search_window：搜索窗口的大小，用於尋找相似區域，默認為21。
    - block_size：块的大小，用於計算相似性，默認為7。

    返回值：
    - image_rgb：RGB格式的原始圖像。
    - deblocked_image：去塊處理後的圖像。
    """

    # 讀取圖像
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"無法打開或找到圖像: {image_path}")

    # 將圖像從 BGR 轉換為 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用非局部均值濾波器進行去塊處理
    deblocked_image = cv2.fastNlMeansDenoisingColored(image_rgb, None, h, h, search_window, block_size)

    return image_rgb, deblocked_image