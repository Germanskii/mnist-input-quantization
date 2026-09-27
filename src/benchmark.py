"""Measure the accuracy and runtime trade-offs of image resizing and pixel quantization in a small PyTorch MLP."""
import argparse
import copy, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from sklearn.metrics import ConfusionMatrixDisplay

SEED = 42
EPOCHS = 12
BATCH_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT = Path('results/mnist')


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def preprocess(images, size, bits=None):
    x = images.float().unsqueeze(1) / 255.0
    if size != 28:
        x = nn.functional.interpolate(x, size=(size, size), mode='bilinear', align_corners=False, antialias=True)
    if bits is not None:
        levels = 2 ** bits - 1
        x = torch.floor(x.clamp(0, 1) * levels) / levels
    return x

class DigitMLP(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(size * size, 48), nn.ReLU(), nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 10))

    def forward(self, x):
        return self.layers(x)

def synchronize():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

def run_variant(name, size, bits, train_raw, test_raw, train_idx, val_idx):
    seed_all()
    x = preprocess(train_raw.data, size, bits)
    xt = preprocess(test_raw.data, size, bits)
    y, yt = (train_raw.targets, test_raw.targets)
    train = DataLoader(TensorDataset(x[train_idx], y[train_idx]), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val = DataLoader(TensorDataset(x[val_idx], y[val_idx]), batch_size=512)
    test = DataLoader(TensorDataset(xt, yt), batch_size=512)
    model = DigitMLP(size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    best_loss, best_state, history = (float('inf'), None, [])
    synchronize()
    start = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for bx, by in train:
            bx, by = (bx.to(DEVICE), by.to(DEVICE))
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val:
                val_loss += nn.functional.cross_entropy(model(bx.to(DEVICE)), by.to(DEVICE), reduction='sum').item()
        val_loss /= len(val_idx)
        history.append({'epoch': epoch + 1, 'train_loss': total_loss / len(train_idx), 'val_loss': val_loss})
        if val_loss < best_loss:
            best_loss, best_state = (val_loss, copy.deepcopy(model.state_dict()))
    synchronize()
    train_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for bx, _ in test:
            predictions.append(model(bx.to(DEVICE)).argmax(1).cpu())
    yp = torch.cat(predictions).numpy()
    batch = xt[:512].to(DEVICE)
    times = []
    with torch.inference_mode():
        for _ in range(10):
            model(batch)
        for _ in range(30):
            synchronize()
            start = time.perf_counter()
            model(batch)
            synchronize()
            times.append(time.perf_counter() - start)
    median_s = float(np.median(times))
    ConfusionMatrixDisplay.from_predictions(yt.numpy(), yp, labels=range(10))
    plt.title(name + ' — весь test')
    save_figure()
    torch.save({'state_dict': {k: v.cpu() for k, v in best_state.items()}, 'input_size': size, 'input_bits': bits, 'seed': SEED}, OUTPUT / f'{name}.pth')
    pd.DataFrame(history).to_csv(OUTPUT / f'{name}_history.csv', index=False)
    return {'variant': name, 'accuracy': float((yp == yt.numpy()).mean()), 'parameters': sum((p.numel() for p in model.parameters())), 'train_seconds': train_seconds, 'inference_images_sec': len(batch) / median_s, 'batch_size': len(batch), 'device': str(DEVICE)}
def save_figure():
    figure = plt.gcf()
    count = len(list(OUTPUT.glob('figure_*.png'))) + 1
    figure.savefig(OUTPUT / f'figure_{count:02d}.png', dpi=150, bbox_inches='tight')
    plt.close(figure)

def main(argv=None):
    parser = argparse.ArgumentParser(description='Measure the accuracy and runtime trade-offs of image resizing and pixel quantization in a small PyTorch MLP.')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('results/mnist'))
    args = parser.parse_args(argv)
    global EPOCHS, BATCH_SIZE, SEED, OUTPUT
    EPOCHS, BATCH_SIZE, SEED, OUTPUT = args.epochs, args.batch_size, args.seed, args.output
    if EPOCHS < 1 or BATCH_SIZE < 1: parser.error('epochs and batch size must be positive')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train_raw=MNIST(str(args.data_root),train=True,download=True)
    test_raw=MNIST(str(args.data_root),train=False,download=True)
    indices=torch.randperm(len(train_raw),generator=torch.Generator().manual_seed(SEED))
    cut=int(0.85*len(indices)); train_idx,val_idx=indices[:cut],indices[cut:]
    rows=[run_variant(name,size,bits,train_raw,test_raw,train_idx,val_idx)
          for name,size,bits in [('baseline_28',28,None),('resized_16',16,None),('quantized_input_16',16,4)]]
    results=pd.DataFrame(rows); print(results.to_string(index=False))
    results.to_csv(OUTPUT/'comparison.csv',index=False)


if __name__ == '__main__':
    main()
