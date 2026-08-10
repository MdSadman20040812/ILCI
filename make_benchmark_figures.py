from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

path = Path(r"D:\github projects\ILCI\data\ilci_benchmark_raw.csv")
out = Path(r"D:\github projects\ILCI\figs")
out.mkdir(exist_ok=True)

rows = []
with open(path, newline='', errors='ignore') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('init') or line == 'DONE':
            continue
        parts = line.split(',')
        if len(parts) != 8:
            continue
        rows.append({
            'init': parts[0],
            'task': parts[1],
            'shape': parts[2],
            'cycles': int(parts[3]),
            'flash': int(parts[4]),
            'epoch': int(parts[5]),
            'loss': None if parts[6].lower() in ('nan', 'inf', '-inf') else float(parts[6]),
            'acc': None if parts[7].lower() in ('nan', 'inf', '-inf') else float(parts[7]),
        })

tasks = ['XOR', 'Iris', 'ECG']
inits = ['StdRandom', 'Gaussian', 'ILCI']
colors = {'StdRandom': '#4C78A8', 'Gaussian': '#F58518', 'ILCI': '#54A24B'}
markers = {'StdRandom': 'o', 'Gaussian': 's', 'ILCI': 'D'}

# figure 1: loss curves
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=220)
for ax, task in zip(axes, tasks):
    for init in inits:
        sub = [r for r in rows if r['task'] == task and r['init'] == init and r['loss'] is not None]
        if not sub:
            continue
        epochs = [r['epoch'] for r in sub]
        vals = [r['loss'] for r in sub]
        ax.plot(epochs, vals, label=init, color=colors[init], marker=markers[init],
                markersize=2.8, linewidth=1.6)
    ax.set_title(f'{task} — Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.grid(True, alpha=0.25)
    if task == 'XOR':
        ax.set_ylim(0.24, 0.34)
axes[1].legend(frameon=True, fontsize=9)
fig.suptitle('ILCI Benchmark: Training Loss Curves on ATmega328P', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(out / 'fig_loss_curves.png', bbox_inches='tight')
plt.close(fig)

# figure 2: accuracy curves
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=220)
for ax, task in zip(axes, tasks):
    for init in inits:
        sub = [r for r in rows if r['task'] == task and r['init'] == init and r['acc'] is not None]
        if not sub:
            continue
        epochs = [r['epoch'] for r in sub]
        vals = [r['acc'] for r in sub]
        ax.plot(epochs, vals, label=init, color=colors[init], marker=markers[init],
                markersize=2.8, linewidth=1.6)
    ax.set_title(f'{task} — Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
axes[1].legend(frameon=True, fontsize=9)
fig.suptitle('ILCI Benchmark: Classification Accuracy on ATmega328P', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(out / 'fig_accuracy_curves.png', bbox_inches='tight')
plt.close(fig)

# summary numbers
summary = {}
for init in inits:
    summary[init] = {}
    for task in tasks:
        sub = [r for r in rows if r['task'] == task and r['init'] == init]
        acc = None
        cyc = None
        if sub:
            cyc = sub[0]['cycles']
            valid = [r['acc'] for r in sub if r['acc'] is not None]
            if valid:
                acc = valid[-1]
        summary[init][task] = {'acc': acc, 'cycles': cyc}

# figure 3: grouped bar summary
fig = plt.figure(figsize=(13, 5.5), dpi=220)
gs = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

x = np.arange(len(tasks))
width = 0.25
for i, init in enumerate(inits):
    accs = [summary[init][t]['acc'] if summary[init][t]['acc'] is not None else 0.0 for t in tasks]
    bars = ax1.bar(x + (i - 1) * width, accs, width, label=init, color=colors[init])
    for b in bars:
        h = b.get_height()
        if h == 0:
            ax1.text(b.get_x() + b.get_width() / 2, 0.02, 'NaN', ha='center', va='bottom', fontsize=8, rotation=90)
        else:
            ax1.text(b.get_x() + b.get_width() / 2, h + 0.01, f'{h:.2f}', ha='center', va='bottom', fontsize=8)
ax1.set_ylabel('Final accuracy')
ax1.set_title('Final accuracy by task')
ax1.set_xticks(x)
ax1.set_xticklabels(tasks)
ax1.set_ylim(0, 1.15)
ax1.legend(frameon=True, fontsize=9)
ax1.grid(True, axis='y', alpha=0.25)

for i, init in enumerate(inits):
    vals = [summary[init][t]['cycles'] if summary[init][t]['cycles'] is not None else 0 for t in tasks]
    bars = ax2.bar(x + (i - 1) * width, vals, width, label=init, color=colors[init])
    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + max(vals) * 0.02, str(int(h)), ha='center', va='bottom', fontsize=8)
ax2.set_ylabel('Cycles / weight')
ax2.set_title('Init cost by task')
ax2.set_xticks(x)
ax2.set_xticklabels(tasks)
ax2.legend(frameon=True, fontsize=9)
ax2.grid(True, axis='y', alpha=0.25)

fig.suptitle('ILCI Benchmark Summary: ATmega328P, 3 Runs', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(out / 'fig_summary_bars.png', bbox_inches='tight')
plt.close(fig)

# figure 4: box plot
fig, ax = plt.subplots(figsize=(10, 4.8), dpi=220)
box_data = []
box_labels = []
for init in inits:
    for task in tasks:
        sub = [r['acc'] for r in rows if r['task'] == task and r['init'] == init and r['acc'] is not None]
        if sub:
            box_data.append(sub)
            box_labels.append(f'{init}\n{task}')
bp = ax.boxplot(box_data, patch_artist=True, showmeans=True, meanline=True)
ax.set_xticklabels(box_labels)
for patch, label in zip(bp['boxes'], box_labels):
    init = label.split('\n')[0]
    patch.set_facecolor(colors[init])
    patch.set_alpha(0.35)
ax.set_ylabel('Accuracy per epoch')
ax.set_title('Accuracy distribution per initialization and task')
ax.grid(True, axis='y', alpha=0.25)
ax.tick_params(axis='x', labelsize=8)
fig.suptitle('ILCI Benchmark: Accuracy Distribution on ATmega328P', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(out / 'fig_accuracy_boxplot.png', bbox_inches='tight')
plt.close(fig)

print('figures written to', out)
