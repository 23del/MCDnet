from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.loveda_dataset import *
from geoseg.models.UNetFormer import UNetFormer
from geoseg.models.DualBranchModel import DualBranchModel
from tools.utils import Lookahead
from tools.utils import process_model_params
# from geoseg.models.vision_mamba import MambaUnet

# training hparam
max_epoch = 80
ignore_index = len(CLASSES)
train_batch_size = 4
val_batch_size = 4
lr = 6e-4
# lr = 0.001
weight_decay = 0.01
backbone_lr = 6e-5
# backbone_lr = 0.001
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "UNetFormer-512crop-epoch80-mbm(cg34)" # 模型权重的文件名
weights_path = "model_weights0/loveda/{}".format(weights_name) # 模型权重保存路径，format()函数用于格式化路径
test_weights_name = "UNetFormer-512crop-epoch80-mbm(cg34)" # 用于测试的权重文件名
log_name = 'loveda/{}'.format(weights_name) # 日志文件保存路径
monitor = 'val_mIoU' # train_mIoU
monitor_mode = 'max'
save_top_k = 1 # 最优模型的数量
save_last = True # 保存最后的模型
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # 预训练模型的路径
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # 继续从之前的检查点恢复训练

#  define the network
# net = DualBranchModel('barren', num_classes=num_classes) #
net = UNetFormer(num_classes=num_classes)
# net = MambaUnet(config, img_size=args.patch_size,
#                     num_classes=num_classes)
# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index) # 忽略的标签索引
use_aux_loss = True # 辅助损失

# define the dataloader

def get_training_transform(): # 数据增强操作，使用albumentations库进行水平翻转
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=ignore_index, nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


train_dataset = LoveDATrainDataset(transform=train_aug, data_root='data/LoveDA/train_val')

val_dataset = loveda_val_dataset

test_dataset = LoveDATestDataset()

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=0,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=0,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)


# Training loop
def train():
    for epoch in range(max_epoch):
        net.train()  # Set the model to training mode
        running_loss = 0.0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)

            optimizer.zero_grad()  # Zero the gradients

            output = net(img)  # Forward pass
            loss_val = loss(output, mask)  # Compute loss

            loss_val.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss_val.item()  # Accumulate loss for the epoch

        print(f"Epoch [{epoch + 1}/{max_epoch}], Loss: {running_loss / len(train_loader)}")

        # Validation step
        if (epoch + 1) % check_val_every_n_epoch == 0:
            validate(epoch)

        # Learning rate scheduler step
        lr_scheduler.step()

        # Save checkpoint
        if save_last:
            torch.save(net.state_dict(), f"{weights_path}/{weights_name}_last.pth")


# Validation function
def validate(epoch):
    net.eval()  # Set the model to evaluation mode
    running_val_loss = 0.0
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            output = net(img)
            loss_val = loss(output, mask)
            running_val_loss += loss_val.item()

    print(f"Validation Epoch [{epoch + 1}], Loss: {running_val_loss / len(val_loader)}")

    # Save the best model based on validation loss (or other metric)
    if monitor == 'val_mIoU' and running_val_loss < monitor_mode:
        torch.save(net.state_dict(), f"{weights_path}/{weights_name}_best.pth")
        print("Best model saved!")


if __name__ == '__main__':
    train()
