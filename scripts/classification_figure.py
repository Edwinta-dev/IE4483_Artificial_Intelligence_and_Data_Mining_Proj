"""Confusion matrix, error types and confidence distribution figure.

Usage: python scripts/classification_figure.py . docs/figures/classification_distribution.png
"""
import csv, glob, sys
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = Path(sys.argv[1]); out = Path(sys.argv[2])
run = Path(glob.glob(str(R/"runs/seeds/aug/seed_1/*/"))[0])
v = list(csv.DictReader(open(run/"val_predictions.csv")))
y = np.array([int(r["true"]) for r in v]); p = np.array([float(r["p_dog"]) for r in v]); yh = (p >= .5).astype(int)
cm = np.array([[np.sum((y==t)&(yh==q)) for q in (0,1)] for t in (0,1)])
conf = list(csv.DictReader(open(R/"runs/report/summary/confusion.csv")))
test = np.array([float(r["p_dog"]) for r in csv.DictReader(open(run/"test_predictions.csv"))])
CAT, DOG = "#3b6fb6", "#d9822b"
fig, ax = plt.subplots(2, 2, figsize=(13, 10)); fig.suptitle("Cats vs Dogs: how the classifications are distributed", fontsize=15, weight="bold")
a = ax[0,0]; a.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        a.text(j, i, f"{cm[i,j]:,}\n({cm[i,j]/cm[i].sum():.2%})", ha="center", va="center", fontsize=13, color="white" if cm[i,j] > 1000 else "black")
a.set_xticks([0,1], ["predicted cat","predicted dog"]); a.set_yticks([0,1], ["true cat","true dog"])
a.set_title(f"A. Confusion matrix, submission model\nResNet-50 head only, augmented data, seed 1 (val, n={len(y):,}, acc {np.mean(y==yh):.2%})")
a = ax[0,1]; lab = [f"{r['backbone'][-2:]} {dict(frozen='head',last_block='last blk',full='full')[r['freeze']]}\n{r['pipeline'][:4]}" for r in conf]
x = np.arange(len(conf)); a.bar(x-.2, [int(r["fp"]) for r in conf], .4, color=CAT, label="cat called dog")
a.bar(x+.2, [int(r["fn"]) for r in conf], .4, color=DOG, label="dog called cat")
a.set_xticks(x, lab, fontsize=7, rotation=90); a.set_ylabel("errors out of 4,999 val images"); a.legend(); a.set_title("B. Error types in all 18 runs (seed 42)\n18/50 = ResNet-18/50; base/augm/mixe = data version")
a = ax[1,0]; bins = np.linspace(0,1,41)
a.hist(p[y==0], bins, color=CAT, alpha=.8, label="true cats"); a.hist(p[y==1], bins, color=DOG, alpha=.8, label="true dogs")
a.axvline(.5, color="k", ls="--", lw=1); a.set_yscale("log"); a.set_xlabel("model's probability the image is a dog"); a.set_ylabel("images (log scale)")
wrong = y != yh; a.set_title(f"C. Confidence on val: almost all near 0 or 1\n{wrong.sum()} wrong; {np.sum((p>.1)&(p<.9))} images between 0.1 and 0.9"); a.legend()
a = ax[1,1]; a.hist(test, bins, color="#6a6a6a"); a.axvline(.5, color="k", ls="--", lw=1); a.set_yscale("log")
nd = int(np.sum(test>=.5)); a.set_title(f"D. Test set (submission, n={len(test)}): {len(test)-nd} cats, {nd} dogs\n{np.sum((test>.1)&(test<.9))} uncertain (between 0.1 and 0.9)")
a.set_xlabel("model's probability the image is a dog"); a.set_ylabel("images (log scale)")
fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=130)
print(cm.tolist(), wrong.sum(), len(test)-nd, nd)
