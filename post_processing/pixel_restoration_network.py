import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# 定義像素恢復網絡模型
class PixelRestorationNetwork(nn.Module):
    def __init__(self):
        super(PixelRestorationNetwork, self).__init__()
        # 定義卷積層
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 前向傳播過程
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# 定義自定義數據集類別
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # 定義數據預處理轉換
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 轉換圖像為Tensor格式
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 正規化圖像
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = self.transform(image)
        return image, image  # 返回原始圖像和目標恢復圖像相同

# 加載訓練數據集
train_data = [...]  # 這裡應該是一個包含訓練數據的列表
train_dataset = CustomDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 初始化模型和優化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PixelRestorationNetwork().to(device)
criterion = nn.MSELoss()  # 定義損失函數
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 定義優化器

# 訓練模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # 設置模型為訓練模式
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  # 梯度清零
        outputs = model(inputs)  # 正向傳播
        loss = criterion(outputs, targets)  # 計算損失
        loss.backward()  # 反向傳播
        optimizer.step()  # 更新模型參數
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# 保存訓練好的模型
torch.save(model.state_dict(), "pixel_restoration_model.pth")

# 加載訓練好的模型權重
model = PixelRestorationNetwork()
model.load_state_dict(torch.load("pixel_restoration_model.pth"))
model.eval()  # 設置為評估模式

# 讀取圖像
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 定義預處理轉換
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # 調整圖像大小為模型輸入大小
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 正規化
])

# 對照片進行預處理
input_image = preprocess(image).unsqueeze(0)  # 添加批次維度

# 使用模型進行修復
with torch.no_grad():
    output_image = model(input_image)

# 將修復後的圖片轉換為PIL圖像
output_image = output_image.squeeze(0).cpu().detach()
output_image = transforms.ToPILImage()(output_image)

# 保存修復後的照片
output_image.save("restored_image.jpg")