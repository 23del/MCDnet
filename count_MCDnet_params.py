import torch
import time
from thop import profile
from geoseg.models.UNetFormer import UNetFormer # 确保你仓库路径正确

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Parameters] Total: {total/1e6:.2f} M, Trainable: {trainable/1e6:.2f} M")

def count_flops(model, input_size=(1, 3, 512, 512)):
    dummy_input = torch.randn(input_size)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"[Complexity] MACs: {macs/1e9:.2f} G, FLOPs≈{2*macs/1e9:.2f} G")

def measure_speed(model, device, input_size=(1, 3, 512, 512), repeat=100):
    dummy_input = torch.randn(input_size).to(device)
    model = model.to(device).eval()

    # warmup
    for _ in range(10):
        _ = model(dummy_input)

    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()
    for _ in range(repeat):
        _ = model(dummy_input)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.time()

    avg_time = (end - start) / repeat
    print(f"[Speed] Avg time: {avg_time*1000:.2f} ms, FPS: {1/avg_time:.2f}")

if __name__ == "__main__":
    # 根据你的任务调整 num_classes 和输入尺寸
    model = UNetFormer(num_classes=7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    count_params(model)
    count_flops(model, input_size=(1, 3, 512, 512))
    measure_speed(model, device, input_size=(1, 3, 512, 512))
