import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

class InputDependentNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1_base = np.random.randn(input_dim, hidden_dim) * 0.5
        self.W2_base = np.random.randn(hidden_dim, output_dim) * 0.5
        self.W1_mod = np.random.randn(input_dim, hidden_dim) * 0.5
        self.W2_mod = np.random.randn(input_dim, output_dim) * 0.5
        # (x + a) * b per dimension
        self.a = np.array([0.5, 0.5])   # offset per dim
        self.b = np.array([1.0, 1.0])   # scale per dim
        self.lr = 0.01
        self.params = ['W1_base', 'W1_mod', 'W2_base', 'W2_mod', 'a', 'b']

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def transform(self, x):
        return (x + self.a) * self.b   # (x+a)*b per dim

    def forward(self, x):
        x_t = self.transform(x)
        mod1 = x_t @ self.W1_mod
        h = np.einsum('bi,ih,bh->bh', x_t, self.W1_base, mod1)
        mod2 = x_t @ self.W2_mod
        out = np.einsum('bh,ho,bo->bo', h, self.W2_base, mod2)
        return out, h, mod1, x_t

    def loss_fn(self, x, y):
        out, *_ = self.forward(x)
        pred = self.sigmoid(out)
        return -np.mean(y*np.log(pred+1e-8)+(1-y)*np.log(1-pred+1e-8))

    def train_step(self, x, y):
        eps = 1e-4
        for name in self.params:
            W = getattr(self, name)
            grad = np.zeros_like(W)
            for idx in np.ndindex(W.shape):
                W[idx] += eps; lp = self.loss_fn(x, y)
                W[idx] -= 2*eps; lm = self.loss_fn(x, y)
                W[idx] += eps
                grad[idx] = (lp - lm) / (2*eps)
            setattr(self, name, W - self.lr * grad)
        out, h, mod1, x_t = self.forward(x)
        pred = self.sigmoid(out)
        loss = -np.mean(y*np.log(pred+1e-8)+(1-y)*np.log(1-pred+1e-8))
        return loss, pred, h, mod1, x_t

net = InputDependentNet(2, 4, 1)
losses, pred_hist, mod1_hist, h_hist, a_hist, b_hist = [], [], [], [], [], []

for epoch in range(3000):
    loss, pred, h, mod1, x_t = net.train_step(X, y)
    if epoch % 50 == 0:
        losses.append(loss)
        pred_hist.append(pred.flatten().copy())
        mod1_hist.append(mod1.copy())
        h_hist.append(h.copy())
        a_hist.append(net.a.copy())
        b_hist.append(net.b.copy())
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: loss={loss:.4f} preds={pred.flatten().round(3)} a={net.a.round(3)} b={net.b.round(3)}")

pred_hist = np.array(pred_hist)
mod1_hist = np.array(mod1_hist)
h_hist = np.array(h_hist)
a_hist = np.array(a_hist)
b_hist = np.array(b_hist)
epochs_s = np.arange(0, 3000, 50)
out_f, h_f, mod1_f, x_t_f = net.forward(X)

print(f"\nFinal a={net.a.round(4)} b={net.b.round(4)}")
print(f"[0,0] maps to: {net.transform(np.array([0.,0.])).round(4)}")
for xi in X:
    print(f"  {xi} -> {net.transform(xi).round(3)}")

# ======= HEATMAPS =======
fig = plt.figure(figsize=(22, 28), facecolor='#0d0d0d')
fig.suptitle('(x+a)×b per dim | Learned affine transform | XOR', fontsize=13, color='white', fontweight='bold', y=0.99)
gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.5, wspace=0.35)
sample_labels = ['[0,0]','[0,1]','[1,0]','[1,1]']

def hm(ax, data, title, cmap='RdBu_r', yticks=None, vmin=None, vmax=None):
    if vmin is None and vmax is None:
        v = np.abs(data).max() + 1e-8
        vmin, vmax = (-v, v) if cmap in ['RdBu_r','coolwarm'] else (None, None)
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, color='white', fontsize=9)
    ax.tick_params(colors='white')
    if yticks:
        ax.set_yticks(range(len(yticks))); ax.set_yticklabels(yticks, color='white', fontsize=8)
    plt.colorbar(im, ax=ax).ax.yaxis.set_tick_params(color='white')

# Row 0
ax = fig.add_subplot(gs[0,0])
ax.plot(epochs_s, losses, color='#ff6b35', lw=2)
ax.fill_between(epochs_s, losses, alpha=0.3, color='#ff6b35')
ax.set_facecolor('#1a1a1a'); ax.set_title('Loss', color='white', fontsize=10)
ax.tick_params(colors='white'); ax.spines[:].set_color('#444')

hm(fig.add_subplot(gs[0,1]), pred_hist.T, 'Predictions Over Time', 'inferno', sample_labels, 0, 1)

ax = fig.add_subplot(gs[0,2])
ax.plot(epochs_s, a_hist[:,0], color='#ff6b35', lw=2, label='a[0]')
ax.plot(epochs_s, a_hist[:,1], color='#44aaff', lw=2, label='a[1]')
ax.set_facecolor('#1a1a1a'); ax.set_title('Learned a (offset) Evolution', color='white', fontsize=10)
ax.tick_params(colors='white'); ax.spines[:].set_color('#444')
ax.legend(facecolor='#333', labelcolor='white', fontsize=8)

ax = fig.add_subplot(gs[0,3])
ax.plot(epochs_s, b_hist[:,0], color='#ff6b35', lw=2, label='b[0]')
ax.plot(epochs_s, b_hist[:,1], color='#44aaff', lw=2, label='b[1]')
ax.set_facecolor('#1a1a1a'); ax.set_title('Learned b (scale) Evolution', color='white', fontsize=10)
ax.tick_params(colors='white'); ax.spines[:].set_color('#444')
ax.legend(facecolor='#333', labelcolor='white', fontsize=8)

# Row 1
hm(fig.add_subplot(gs[1,0]), mod1_hist.reshape(len(epochs_s),-1).T, 'Modulator Evolution', 'plasma')
hm(fig.add_subplot(gs[1,1]), h_hist.reshape(len(epochs_s),-1).T, 'Hidden Evolution', 'RdBu_r')
hm(fig.add_subplot(gs[1,2]), mod1_f, 'Final Modulators\n(samples × neurons)', 'plasma', sample_labels)
hm(fig.add_subplot(gs[1,3]), h_f, 'Final Hidden\n(samples × neurons)', 'RdBu_r', sample_labels)

# Row 2
trans_labels = [f'{s}→{net.transform(X[i]).round(2)}' for i,s in enumerate(sample_labels)]
for idx in range(4):
    eff = np.outer(np.ones(2), mod1_f[idx]) * net.W1_base
    hm(fig.add_subplot(gs[2,idx]), eff, f'Effective W1\n{trans_labels[idx]}', 'RdBu_r')

# Row 3
for pi, (i,j,label) in enumerate([(0,3,'[0,0] vs [1,1]'),(1,2,'[0,1] vs [1,0]')]):
    eff_i = np.outer(np.ones(2), mod1_f[i]) * net.W1_base
    eff_j = np.outer(np.ones(2), mod1_f[j]) * net.W1_base
    hm(fig.add_subplot(gs[3,pi*2:pi*2+2]), eff_j-eff_i, f'Effective W1 diff: {label}', 'RdBu_r')

ax = fig.add_subplot(gs[3,2:4])
colors = ['#ff4444','#44ff44','#44ff44','#ff4444']
for i, (point, col) in enumerate(zip(x_t_f, colors)):
    ax.scatter(point[0], point[1], c=col, s=400, zorder=5, edgecolors='white', lw=1.5)
    ax.annotate(f'{trans_labels[i]}\ny={int(y[i][0])}', point,
                textcoords='offset points', xytext=(8,5), color='white', fontsize=8)
ax.set_facecolor('#1a1a1a')
ax.set_title(f'Learned Transform Space\na={net.a.round(3)} b={net.b.round(3)}', color='white', fontsize=10)
ax.set_xlabel('dim 0', color='#aaa'); ax.set_ylabel('dim 1', color='#aaa')
ax.tick_params(colors='white'); ax.spines[:].set_color('#444'); ax.grid(alpha=0.2, color='white')

plt.savefig('xor_affine_ab.png', dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
print("saved")