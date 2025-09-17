import numpy as np
import matplotlib.pyplot as plt


vals  = np.array([1, 0, -1])
probs = np.array([0.5, 0.25, 0.25])
log2p = {1: np.log2(0.5), 0: np.log2(0.25), -1: np.log2(0.25)}

rng = np.random.default_rng(1)  

def sample_seq(N):
    return rng.choice(vals, size=N, p=probs)

def Hhat_bits(seq):
    return - np.mean([log2p[int(x)] for x in seq])

Ns = [10, 50, 100, 500, 1000]
hhats = []

for N in Ns:
    seq = sample_seq(N)
    hhats.append(Hhat_bits(seq))
    c1 = (seq==1).sum(); c0 = (seq==0).sum(); cm1 = (seq==-1).sum()
    print(f"N={N:4d} , distribution: 1:{c1:4d} 0:{c0:4d} -1:{cm1:4d} , H'={hhats[-1]:.4f} bits")

H_true = 1.5 

plt.plot(Ns, hhats, "o-", label=r"$H'(N)$ (bits)")
plt.axhline(H_true, ls="--", label=r"$H(X)=1.5$ bits")
plt.xlabel("N"); plt.ylabel("Entropy (bits)"); plt.title(r"$H'(N)$")
path = "Hhat_vs_N.png"
plt.savefig(path, dpi=300, bbox_inches="tight")

plt.legend(); plt.tight_layout(); plt.show()
