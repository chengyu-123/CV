import numpy as np
import cv2

# 定義區塊匹配函數
def block_matching(prev_frame, next_frame, block_size=16, search_range=16):

    """
    尋找兩幀之間的運動向量。

    參數：
    - prev_frame：前一幀（參考幀）的灰度圖像。
    - next_frame：下一幀的灰度圖像。
    - block_size：區塊的大小，默認為16x16。
    - search_range：搜索範圍，默認為16。

    返回值：
    - motion_vectors：每個區塊的運動向量，以(x, y)形式表示。
    """

    height, width = prev_frame.shape[:2]
    # 初始化運動向量矩陣
    motion_vectors = np.zeros((height // block_size, width // block_size, 2), dtype=np.int32)

    # 進行區塊匹配
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # 提取當前區塊
            block_prev = prev_frame[i:i+block_size, j:j+block_size]
            best_match = None
            min_sad = float('inf')

            # 在搜索範圍內尋找最佳匹配
            for m in range(max(0, i-search_range), min(height-block_size, i+search_range)):
                for n in range(max(0, j-search_range), min(width-block_size, j+search_range)):
                    # 提取參考幀中的區塊
                    block_next = next_frame[m:m+block_size, n:n+block_size]
                    # 計算SAD（總絕對差）
                    sad = np.sum(np.abs(block_prev - block_next))

                    # 更新最小SAD和最佳匹配
                    if sad < min_sad:
                        min_sad = sad
                        best_match = (m-i, n-j)

            # 將最佳匹配的運動向量存入運動向量矩陣
            motion_vectors[i//block_size, j//block_size] = best_match

    return motion_vectors

# 定義運動補償函數
def motion_compensation(prev_frame, motion_vectors, block_size=16):

    """
    使用運動向量進行運動補償。

    參數：
    - prev_frame：前一幀（參考幀）的灰度圖像。
    - motion_vectors：每個區塊的運動向量。
    - block_size：區塊的大小，默認為16x16。

    返回值：
    - compensated_frame：補償後的當前幀圖像。
    """

    height, width = prev_frame.shape[:2]
    # 初始化補償後的幀
    compensated_frame = np.zeros_like(prev_frame)

    # 遍歷每個區塊，進行運動補償
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # 獲取該區塊的運動向量
            mv = motion_vectors[i//block_size, j//block_size]
            # 根據運動向量進行補償
            x, y = j + mv[1], i + mv[0]
            compensated_frame[i:i+block_size, j:j+block_size] = prev_frame[y:y+block_size, x:x+block_size]

    return compensated_frame