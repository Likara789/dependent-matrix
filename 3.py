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
N_SHAPES  = 10
SHAPE_DIM = 1
LOOPS     = 5

class ThinkingNet(nn.Module):
    def __init__(self, input_dim=784, n_shapes=N_SHAPES, shape_dim=SHAPE_DIM):
        super().__init__()
        self.n_shapes  = n_shapes
        self.shape_dim = shape_dim
        h_dim          = n_shapes * shape_dim

        # input pathway: x modulates the state transition
        self.W_state   = nn.Linear(h_dim, h_dim)
        self.W_inp_mod = nn.Linear(input_dim, h_dim, bias=False)
        self.norm      = nn.LayerNorm(h_dim)

        # readout from final shapes
        self.head = nn.Linear(h_dim, 10)

    def step(self, x, h):
        # x steers how h evolves — never concatenated
        mod   = self.W_inp_mod(x)           # (B, h_dim)  — input gate
        h_new = F.relu(self.W_state(h) * mod)
        return self.norm(h_new)

    def forward(self, x):
        B = x.size(0)
        h = torch.zeros(B, self.n_shapes * self.shape_dim, device=x.device)
        hs = []
        for _ in range(LOOPS):
            h = self.step(x, h)
            hs.append(h)
        out = self.head(h)
        return out, hs

# ── training ──────────────────────────────────────────────────────────────────
net = ThinkingNet().to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")

train_losses, train_accs, test_accs = [], [], []

def accuracy(dl):
    correct = total = 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            out, _ = net(x)
            correct += (out.argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct / total

for ep in range(20):
    net.train()
    ep_loss = 0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out, _ = net(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        ep_loss += loss.item()
    ep_loss /= len(train_dl)
    tr_acc = accuracy(train_dl)
    te_acc = accuracy(test_dl)
    train_losses.append(ep_loss)
    train_accs.append(tr_acc)
    test_accs.append(te_acc)

    # dead neuron count across all shapes
    net.eval()
    with torch.no_grad():
        all_h = torch.cat([net(x.to(device))[1][-1] for x, _ in test_dl])
    dead = (all_h.mean(0) == 0).sum().item()
    total_neurons = N_SHAPES * SHAPE_DIM
    net.train()
    print(f"Ep {ep+1:2d}: loss={ep_loss:.4f}  train={tr_acc:.3f}  test={te_acc:.3f}  dead={dead}/{total_neurons}")

# ── thinking loop analysis ─────────────────────────────────────────────────────
print("\nAnalyzing thinking loops...")
net.eval()

# track how shapes evolve per class across loops
class_shapes_per_loop = []  # (LOOPS, 10, n_shapes*shape_dim)
with torch.no_grad():
    for loop_idx in range(LOOPS):
        loop_class = []
        for c in range(10):
            xs = torch.stack([x for x, y in test_ds if y == c])[:200].to(device)
            _, hs = net(xs)
            loop_class.append(hs[loop_idx].cpu().numpy().mean(axis=0))
        class_shapes_per_loop.append(np.array(loop_class))  # (10, n_shapes*shape_dim)

class_shapes_per_loop = np.array(class_shapes_per_loop)  # (LOOPS, 10, n_shapes*shape_dim)

# reshape to (LOOPS, 10, n_shapes, shape_dim) and take norm per shape slot
shaped = class_shapes_per_loop.reshape(LOOPS, 10, N_SHAPES, SHAPE_DIM)
shape_norms = np.linalg.norm(shaped, axis=-1)  # (LOOPS, 10, n_shapes)

# ── visualisation ─────────────────────────────────────────────────────────────
digit_labels = [str(i) for i in range(10)]
shape_labels = [f"s{i}" for i in range(N_SHAPES)]

fig = plt.figure(figsize=(24, 28), facecolor='#0d0d0d')
fig.suptitle('ThinkingNet — Input-Dependent Looped Reasoning on MNIST',
             fontsize=14, color='white', fontweight='bold', y=0.99)
gs = gridspec.GridSpec(4, 5, figure=fig, hspace=0.5, wspace=0.4)

def styled_ax(ax, title):
    ax.set_facecolor('#1a1a1a')
    ax.set_title(title, color='white', fontsize=9)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')

def hm(ax, data, title, cmap='plasma', yticks=None, xticks=None, vmin=None, vmax=None):
    if vmin is None:
        if cmap in ['RdBu_r', 'coolwarm']:
            v = np.abs(data).max() + 1e-8; vmin, vmax = -v, v
        else:
            vmin, vmax = 0, data.max() + 1e-8
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, color='white', fontsize=8)
    ax.tick_params(colors='white', labelsize=7)
    if yticks: ax.set_yticks(range(len(yticks))); ax.set_yticklabels(yticks, color='white', fontsize=7)
    if xticks: ax.set_xticks(range(len(xticks))); ax.set_xticklabels(xticks, color='white', fontsize=7, rotation=90)
    cb = plt.colorbar(im, ax=ax); cb.ax.tick_params(labelcolor='white', labelsize=6)

# Row 0: training curves
ax = fig.add_subplot(gs[0, 0])
ax.plot(train_losses, color='#ff6b35', lw=2)
styled_ax(ax, 'Train Loss')

ax = fig.add_subplot(gs[0, 1])
ax.plot(train_accs, color='#44ff88', lw=2, label='train')
ax.plot(test_accs,  color='#44aaff', lw=2, label='test')
styled_ax(ax, 'Accuracy')
ax.legend(facecolor='#333', labelcolor='white', fontsize=8)

# Row 0: shape norms at each loop step for a sample class (digit 0)
ax = fig.add_subplot(gs[0, 2:4])
for c in range(10):
    norms_over_loops = shape_norms[:, c, :].mean(axis=1)  # mean over shapes
    ax.plot(range(1, LOOPS+1), norms_over_loops, label=str(c), lw=1.5)
styled_ax(ax, 'Mean Shape Norm per Class across Loops')
ax.set_xlabel('loop', color='#aaa')
ax.legend(facecolor='#333', labelcolor='white', fontsize=7, ncol=2)

# Row 1-2: shape norms (digit x shape_slot) at each loop
for loop_idx in range(LOOPS):
    row = 1 + loop_idx // 3
    col = loop_idx % 3 + (2 if loop_idx >= 3 else 0)
    # just lay them out in rows 1 and 2
    ax = fig.add_subplot(gs[1, loop_idx] if loop_idx < 5 else gs[2, loop_idx-5])
    hm(ax, shape_norms[loop_idx],
       f'Shape Norms — Loop {loop_idx+1}\n(rows=digits, cols=shape slots)',
       cmap='plasma', yticks=digit_labels, xticks=shape_labels)

# Row 2: difference between last and first loop (what thinking changed)
delta = shape_norms[-1] - shape_norms[0]
hm(fig.add_subplot(gs[2, 0:2]),
   delta, 'Shape Norm Delta (loop5 - loop1)\n(what thinking changed)',
   cmap='RdBu_r', yticks=digit_labels, xticks=shape_labels)

hm(fig.add_subplot(gs[2, 2:4]),
   shape_norms[-1],
   'Final Shape Norms (loop 5)\n(rows=digits, cols=shape slots)',
   cmap='plasma', yticks=digit_labels, xticks=shape_labels)

# Row 3: convergence — how much h changes between consecutive loops per class
convergence = []
for loop_idx in range(1, LOOPS):
    delta_h = np.abs(class_shapes_per_loop[loop_idx] - class_shapes_per_loop[loop_idx-1]).mean(axis=1)
    convergence.append(delta_h)
convergence = np.array(convergence)  # (LOOPS-1, 10)

ax = fig.add_subplot(gs[3, 0:2])
for c in range(10):
    ax.plot(range(2, LOOPS+1), convergence[:, c], label=str(c), lw=1.5)
styled_ax(ax, 'Mean |Δh| between loops per class\n(convergence speed)')
ax.set_xlabel('loop', color='#aaa')
ax.legend(facecolor='#333', labelcolor='white', fontsize=7, ncol=2)

# W_inp_mod receptive fields per shape slot (input -> h modulator)
W = net.W_inp_mod.weight.detach().cpu().numpy()  # (h_dim, 784)
# mean over shape_dim for each shape slot -> (n_shapes, 784)
W_per_shape = W.reshape(N_SHAPES, SHAPE_DIM, 784).mean(axis=1)

gs3 = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=gs[3, 2:], hspace=0.1, wspace=0.1)
for i in range(N_SHAPES):
    ax = fig.add_subplot(gs3[i//5, i%5])
    rf = W_per_shape[i].reshape(28, 28)
    v  = np.abs(rf).max()
    ax.imshow(rf, cmap='RdBu_r', vmin=-v, vmax=v)
    ax.set_title(f'shape {i}', color='white', fontsize=6)
    ax.axis('off')

plt.savefig('thinking_net.png', dpi=130, bbox_inches='tight', facecolor='#0d0d0d')
print("Saved thinking_net.png")
