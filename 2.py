import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")

# ── data ──────────────────────────────────────────────────────────────────────
tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_ds = datasets.MNIST('.', train=True,  download=True, transform=tf)
test_ds  = datasets.MNIST('.', train=False, download=True, transform=tf)
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=512)

# ── model ─────────────────────────────────────────────────────────────────────
class InputDepNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.W1_base = nn.Linear(input_dim, hidden_dim)
        self.W1_mod  = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W2_base = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        mod1 = self.W1_mod(x)
        h    = F.relu(self.W1_base(x) * mod1)
        out  = self.W2_base(h)
        return out, h, mod1

# ── training ──────────────────────────────────────────────────────────────────
net = InputDepNet(hidden_dim=3).to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

train_losses, train_accs, test_accs = [], [], []

def accuracy(dl):
    correct = total = 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            out, *_ = net(x)
            correct += (out.argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct / total

for ep in range(20):
    net.train()
    ep_loss = 0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out, *_ = net(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step()
        ep_loss += loss.item()
    ep_loss /= len(train_dl)
    tr_acc = accuracy(train_dl)
    te_acc = accuracy(test_dl)
    train_losses.append(ep_loss)
    train_accs.append(tr_acc)
    test_accs.append(te_acc)
    # dead neuron count: neurons where mean activation across test set is 0
    net.eval()
    with torch.no_grad():
        all_h = torch.cat([net(x.to(device))[1] for x, _ in test_dl])
    dead = (all_h.mean(0) == 0).sum().item()
    net.train()
    print(f"Ep {ep+1:2d}: loss={ep_loss:.4f}  train={tr_acc:.3f}  test={te_acc:.3f}  dead={dead}/3")

# ── superposition analysis ────────────────────────────────────────────────────
print("\nRunning superposition analysis...")
net.eval()

class_mods = []
class_acts = []

with torch.no_grad():
    for c in range(10):
        xs = torch.stack([x for x, y in test_ds if y == c])[:200].to(device)
        _, h, mod1 = net(xs)
        class_mods.append(mod1.cpu().numpy().mean(axis=0))
        class_acts.append(h.cpu().numpy().mean(axis=0))

class_mods = np.array(class_mods)   # (10, 256)
class_acts = np.array(class_acts)

thresh = class_acts.max() * 0.3
classes_per_neuron = (class_acts > thresh).sum(axis=0)
mod_variance       = class_mods.var(axis=0)
poly_score         = classes_per_neuron * mod_variance
top_poly           = np.argsort(poly_score)[::-1][:16]

print(f"Dead (0 classes):       {(classes_per_neuron==0).sum()}")
print(f"Monosemantic (1):       {(classes_per_neuron==1).sum()}")
print(f"Low poly (2-4):         {((classes_per_neuron>=2)&(classes_per_neuron<=4)).sum()}")
print(f"High poly (5+):         {(classes_per_neuron>=5).sum()}")

# ── visualisation ─────────────────────────────────────────────────────────────
digit_labels = [str(i) for i in range(10)]

fig = plt.figure(figsize=(24, 26), facecolor='#0d0d0d')
fig.suptitle('Input-Dependent Net on MNIST — Superposition Analysis',
             fontsize=14, color='white', fontweight='bold', y=0.99)
gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.5, wspace=0.4)

def hm(ax, data, title, cmap='RdBu_r', xticks=None, yticks=None, vmin=None, vmax=None):
    if vmin is None and vmax is None:
        v = np.abs(data).max() + 1e-8
        vmin, vmax = (-v, v) if cmap in ['RdBu_r', 'coolwarm'] else (0, data.max()+1e-8)
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, color='white', fontsize=9)
    ax.tick_params(colors='white', labelsize=7)
    if yticks: ax.set_yticks(range(len(yticks))); ax.set_yticklabels(yticks, color='white', fontsize=7)
    if xticks: ax.set_xticks(range(len(xticks))); ax.set_xticklabels(xticks, color='white', fontsize=7, rotation=90)
    cb = plt.colorbar(im, ax=ax); cb.ax.yaxis.set_tick_params(color='white')
    cb.ax.tick_params(labelcolor='white', labelsize=6)

# Row 0: training curves + histograms
ax = fig.add_subplot(gs[0,0])
ax.plot(train_losses, color='#ff6b35', lw=2)
ax.set_facecolor('#1a1a1a'); ax.set_title('Train Loss', color='white')
ax.tick_params(colors='white'); ax.spines[:].set_color('#444')

ax = fig.add_subplot(gs[0,1])
ax.plot(train_accs, color='#44ff88', lw=2, label='train')
ax.plot(test_accs,  color='#44aaff', lw=2, label='test')
ax.set_facecolor('#1a1a1a'); ax.set_title('Accuracy', color='white')
ax.tick_params(colors='white'); ax.spines[:].set_color('#444')
ax.legend(facecolor='#333', labelcolor='white', fontsize=8)

ax = fig.add_subplot(gs[0,2])
counts = [(classes_per_neuron==k).sum() for k in range(11)]
ax.bar(range(11), counts, color='#aa44ff')
ax.set_facecolor('#1a1a1a'); ax.set_title('Classes per Neuron\n(activation polysemanticity)', color='white', fontsize=9)
ax.tick_params(colors='white'); ax.spines[:].set_color('#444')
ax.set_xlabel('# classes', color='#aaa'); ax.set_ylabel('# neurons', color='#aaa')

ax = fig.add_subplot(gs[0,3])
ax.hist(mod_variance, bins=40, color='#ff44aa', edgecolor='none')
ax.set_facecolor('#1a1a1a'); ax.set_title('Modulator Variance\nacross classes (per neuron)', color='white', fontsize=9)
ax.tick_params(colors='white'); ax.spines[:].set_color('#444')
ax.set_xlabel('variance', color='#aaa')

# Row 1: per-class modulators and activations
hm(fig.add_subplot(gs[1,0:2]), class_mods,
   'Mean Modulator (mod1) per Class\n(rows=digits 0-9, cols=neurons)',
   'RdBu_r', yticks=digit_labels)

hm(fig.add_subplot(gs[1,2:4]), class_acts,
   'Mean Activation (h_a) per Class\n(rows=digits 0-9, cols=neurons)',
   'plasma', yticks=digit_labels, vmin=0, vmax=class_acts.max())

# Row 2: top polysemantic neurons
hm(fig.add_subplot(gs[2,0:2]), class_mods[:, top_poly],
   'Top-16 Polysemantic Neurons: Modulator per Class',
   'RdBu_r', xticks=[str(n) for n in top_poly], yticks=digit_labels)

hm(fig.add_subplot(gs[2,2:4]), class_acts[:, top_poly],
   'Top-16 Polysemantic Neurons: Activation per Class',
   'plasma', xticks=[str(n) for n in top_poly], yticks=digit_labels, vmin=0)

# Row 3: W1_base receptive fields for top polysemantic neurons
W1 = net.W1_base.weight.detach().cpu().numpy()   # (256, 784)
gs3 = gridspec.GridSpecFromSubplotSpec(2, 8, subplot_spec=gs[3,:], hspace=0.1, wspace=0.1)
for plot_i, neuron_idx in enumerate(top_poly[:16]):
    ax = fig.add_subplot(gs3[plot_i//8, plot_i%8])
    rf = W1[neuron_idx].reshape(28, 28)
    v  = np.abs(rf).max()
    ax.imshow(rf, cmap='RdBu_r', vmin=-v, vmax=v)
    ax.set_title(f'n{neuron_idx}', color='white', fontsize=6)
    ax.axis('off')

fig.text(0.5, 0.02, 'W1_base receptive fields — top-16 polysemantic neurons',
         color='#aaa', fontsize=9, ha='center')

plt.savefig('mnist_superposition.png', dpi=130, bbox_inches='tight', facecolor='#0d0d0d')
print("Saved mnist_superposition.png")
