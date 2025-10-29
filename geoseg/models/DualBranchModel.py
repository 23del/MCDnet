import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# 深度卷积网络（专注于低准确率类别特征提取）
class ConvBranch(nn.Module):
    def __init__(self, low_accuracy_class, num_classes):
        super(ConvBranch, self).__init__()
        self.low_accuracy_class = low_accuracy_class  # 低准确率类别
        self.resnet = models.resnet50(pretrained=True)  # 使用ResNet50作为示例
        self.resnet.fc = nn.Identity()  # 去掉ResNet的全连接层
        self.conv_layers = nn.ModuleList([  # 添加卷积层
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, num_classes, kernel_size=3, padding=1)
        ])

    def forward(self, x):
        features = self.resnet(x)  # 获取ResNet的特征
        for conv in self.conv_layers:  # 通过卷积层
            features = F.relu(conv(features))

        # 提取低准确率类别的特征
        low_accuracy_features = features[:, self.low_accuracy_class, :, :]
        return low_accuracy_features


# Transformer分支（进行语义分割任务）
class TransformerBranch(nn.Module):
    def __init__(self, num_classes, num_layers=4):
        super(TransformerBranch, self).__init__()
        self.num_layers = num_layers
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=2048, nhead=8, dim_feedforward=2048
            ) for _ in range(num_layers)
        ])
        self.classifier = nn.Conv2d(2048, num_classes, kernel_size=3, padding=1)

    def forward(self, x, low_accuracy_features, low_accuracy_class):
        # 假设x为输入的图像特征
        for i in range(self.num_layers):
            x = self.transformer_blocks[i](x)

        # 将卷积提取的低准确率类别特征融合到Transformer特征中
        x[:, low_accuracy_class, :, :] = x[:, low_accuracy_class, :, :] + low_accuracy_features
        return self.classifier(x)


# 双分支网络模型
class DualBranchModel(nn.Module):
    def __init__(self, low_accuracy_class, num_classes):
        super(DualBranchModel, self).__init__()
        self.conv_branch = ConvBranch(low_accuracy_class, num_classes)
        self.transformer_branch = TransformerBranch(num_classes)

    def forward(self, x):
        # 获取低准确率类别的特征
        low_accuracy_features = self.conv_branch(x)

        # Transformer分支的输入
        transformer_input = low_accuracy_features  # Transformer输入使用卷积特征

        output = self.transformer_branch(transformer_input, low_accuracy_features, self.conv_branch.low_accuracy_class)
        return output


# 测试代码
if __name__ == '__main__':
    low_accuracy_class = 10  # 假设类别10是低准确率类别
    model = DualBranchModel(low_accuracy_class=low_accuracy_class, num_classes=701)  # 假设有701类
    input_tensor = torch.randn(1, 3, 384, 384)  # 假设输入图像为384x384
    output = model(input_tensor)
    print(output.shape)  # 输出形状应为 (1, num_classes, 384, 384)
