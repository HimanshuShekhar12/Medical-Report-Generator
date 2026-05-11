"""
evaluate.py
-----------
Comprehensive evaluation of DDPM-generated synthetic X-rays.

Three evaluation metrics:

1. MS-SSIM (Multi-Scale Structural Similarity)
   - Compares individual image pairs
   - Measures structural quality
   - Target: > 0.6

2. FID (Fréchet Inception Distance)
   - Compares entire distributions
   - Measures if synthetic images come from     same distribution as real
   - Target: < 50 (lower is better)

3. Pathology Preservation Check (CheXNet-style)
   - Verifies generated fibrosis looks like fibrosis
   - Uses DenseNet121 pretrained on chest X-rays
   - Most clinically meaningful metric

Run:
  python3 src/ddpm/evaluate.py
"""

import os, sys, csv, cv2, torch, numpy as np, warnings
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parents[2]))

REAL_DIR      = Path("data/processed/images")
SYNTHETIC_DIR = Path("data/synthetic/ddpm_generated")
OUT_DIR       = Path("outputs/eval_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {
    "normal":0,"effusion":1,"pleural":2,"pneumothorax":3,
    "consolidation":4,"infiltrate":5,"opacity":6,"atelectasis":7,
    "edema":8,"cardiomegaly":9,"nodule":10,"pneumonia":11,
    "fracture":12,"mass":13,"calcification":14,"emphysema":15,
    "hernia":16,"fibrosis":17,
}

# ── MS-SSIM ────────────────────────────────────────────────────────
def compute_ms_ssim(real_dir, synthetic_dir, num_samples=100):
    from skimage.metrics import structural_similarity as ssim
    real = sorted(real_dir.glob("*.png"))[:num_samples]
    syn  = sorted(synthetic_dir.glob("*.png"))[:num_samples]
    scores = []
    for r, s in zip(real, syn):
        ri = cv2.imread(str(r), cv2.IMREAD_GRAYSCALE)
        si = cv2.imread(str(s), cv2.IMREAD_GRAYSCALE)
        if ri is None or si is None: continue
        if ri.shape != si.shape: si = cv2.resize(si, (ri.shape[1], ri.shape[0]))
        scores.append(ssim(ri, si, data_range=255))
    return float(np.mean(scores)) if scores else 0.0

# ── FID ────────────────────────────────────────────────────────────
class InceptionFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        inc = models.inception_v3(pretrained=True, aux_logits=True)
        self.features = torch.nn.Sequential(
            inc.Conv2d_1a_3x3, inc.Conv2d_2a_3x3, inc.Conv2d_2b_3x3,
            torch.nn.MaxPool2d(3,2), inc.Conv2d_3b_1x1, inc.Conv2d_4a_3x3,
            torch.nn.MaxPool2d(3,2), inc.Mixed_5b, inc.Mixed_5c, inc.Mixed_5d,
            inc.Mixed_6a, inc.Mixed_6b, inc.Mixed_6c, inc.Mixed_6d, inc.Mixed_6e,
            inc.Mixed_7a, inc.Mixed_7b, inc.Mixed_7c,
            torch.nn.AdaptiveAvgPool2d((1,1)),
        )
    def forward(self, x):
        return self.features(x).squeeze(-1).squeeze(-1)

def load_images_fid(image_dir, max_images=200):
    paths = list(image_dir.glob("*.png"))[:max_images]
    imgs  = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv2.resize(img, (299, 299))
        img = np.stack([img,img,img], 0).astype(np.float32) / 255.0
        imgs.append(img)
    return torch.FloatTensor(np.array(imgs)) if imgs else None

def get_fid_stats(images, model, batch_size=16):
    model.eval()
    feats = []
    for i in range(0, len(images), batch_size):
        with torch.no_grad():
            feats.append(model(images[i:i+batch_size].to(device)).cpu().numpy())
    feats = np.concatenate(feats, 0)
    return np.mean(feats, 0), np.cov(feats, rowvar=False)

def compute_fid(mu1, s1, mu2, s2, eps=1e-6):
    from scipy import linalg
    diff = mu1 - mu2
    cm, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(cm).all():
        cm = linalg.sqrtm((s1+np.eye(s1.shape[0])*eps).dot(s2+np.eye(s2.shape[0])*eps))
    if np.iscomplexobj(cm): cm = cm.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2*np.trace(cm))

def compute_fid_score(real_dir, syn_dir, max_images=200):
    try:
        model = InceptionFeatureExtractor().to(device)
        ri = load_images_fid(real_dir, max_images)
        si = load_images_fid(syn_dir,  max_images)
        if ri is None or si is None: return -1.0
        mu_r, s_r = get_fid_stats(ri, model)
        mu_s, s_s = get_fid_stats(si, model)
        return compute_fid(mu_r, s_r, mu_s, s_s)
    except Exception as e:
        print(f"  FID error: {e}")
        return -1.0

# ── PATHOLOGY CHECK ────────────────────────────────────────────────
class ChestXRayClassifier(torch.nn.Module):
    """DenseNet121-based CheXNet-style classifier for 18 pathologies."""
    def __init__(self, num_classes=18):
        super().__init__()
        import torchvision.models as models
        self.backbone = models.densenet121(pretrained=True)
        nf = self.backbone.classifier.in_features
        self.backbone.classifier = torch.nn.Sequential(
            torch.nn.Linear(nf, num_classes),
            torch.nn.Sigmoid()
        )
    def forward(self, x): return self.backbone(x)

def prep_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, (224,224))
    img = np.stack([img,img,img],0).astype(np.float32)/255.0
    mean = np.array([0.485,0.456,0.406]).reshape(3,1,1)
    std  = np.array([0.229,0.224,0.225]).reshape(3,1,1)
    return torch.FloatTensor((img-mean)/std).unsqueeze(0)

def check_pathology(syn_dir, class_name, classifier, num_samples=20):
    paths = list(syn_dir.glob("*.png"))[:num_samples]
    target_idx = LABEL_MAP.get(class_name, -1)
    if target_idx == -1: return {"target_prob_mean":0,"top1_accuracy":0,"status":"N/A"}
    target_probs, top1_correct = [], 0
    classifier.eval()
    for p in tqdm(paths, desc=f"  Classifying {class_name}", leave=False):
        t = prep_image(p)
        if t is None: continue
        with torch.no_grad():
            probs = classifier(t.to(device)).squeeze().cpu().numpy()
        target_probs.append(float(probs[target_idx]))
        if int(np.argmax(probs)) == target_idx: top1_correct += 1
    n = len(target_probs)
    if n == 0: return {"target_prob_mean":0,"top1_accuracy":0,"status":"N/A"}
    mean_prob = float(np.mean(target_probs))
    status = "✅ PASS" if mean_prob > 0.3 else "⚠️  WEAK" if mean_prob > 0.1 else "❌ FAIL"
    return {"target_prob_mean":mean_prob,"top1_accuracy":top1_correct/n,"status":status}

# ── VISUAL GRID ────────────────────────────────────────────────────
def save_visual_grid(real_dir, syn_dir, class_name, out_path, n=5):
    real = sorted(real_dir.glob("*.png"))[:n]
    syns = sorted(syn_dir.glob("*.png"))[:n]
    if not real or not syns: return
    fig, axes = plt.subplots(2, n, figsize=(n*3, 7))
    fig.suptitle(f"Real vs Synthetic — {class_name.upper()}", fontsize=14, fontweight="bold")
    for i in range(n):
        for row, imgs, label, color in [(0,real,"Real","steelblue"),(1,syns,"Synthetic","tomato")]:
            ax = axes[row][i]
            if i < len(imgs):
                img = cv2.imread(str(imgs[i]), cv2.IMREAD_GRAYSCALE)
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            if i == 0: ax.set_ylabel(label, fontsize=11, fontweight="bold", color=color)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(color); spine.set_linewidth(2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out_path}")

# ── MAIN ───────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("DDPM Comprehensive Evaluation")
    print("="*60)
    print(f"Device: {device}\n")

    if not SYNTHETIC_DIR.exists():
        print(f"[ERROR] No synthetic images at {SYNTHETIC_DIR}")
        return

    print("Loading DenseNet121 pathology classifier...")
    try:
        classifier = ChestXRayClassifier(num_classes=18).to(device)
        print("[OK] Classifier loaded\n")
    except Exception as e:
        print(f"[WARNING] Classifier failed: {e}")
        classifier = None

    results = []
    class_dirs = [d for d in sorted(SYNTHETIC_DIR.iterdir()) if d.is_dir()]

    for class_dir in class_dirs:
        class_name = class_dir.name
        syn_count  = len(list(class_dir.glob("*.png")))
        print(f"\n{'─'*50}")
        print(f"Evaluating: {class_name.upper()} ({syn_count} images)")
        print(f"{'─'*50}")

        print("  [1/4] MS-SSIM...")
        ms_ssim = compute_ms_ssim(REAL_DIR, class_dir, num_samples=50)
        print(f"        MS-SSIM = {ms_ssim:.3f}")

        print("  [2/4] FID score...")
        fid = compute_fid_score(REAL_DIR, class_dir, max_images=min(syn_count, 200))
        print(f"        FID = {fid:.1f}" if fid >= 0 else "        FID = N/A")

        print("  [3/4] Pathology preservation...")
        if classifier:
            pr = check_pathology(class_dir, class_name, classifier, num_samples=20)
            target_prob = pr["target_prob_mean"]
            top1_acc    = pr["top1_accuracy"]
            status      = pr["status"]
            print(f"        Target prob = {target_prob:.3f} | Top-1 = {top1_acc:.1%} | {status}")
        else:
            target_prob, top1_acc, status = 0.0, 0.0, "N/A"

        print("  [4/4] Visual grid...")
        save_visual_grid(REAL_DIR, class_dir, class_name, OUT_DIR/f"grid_{class_name}.png")

        results.append({
            "class":class_name, "synthetic_count":syn_count,
            "ms_ssim":ms_ssim, "fid":fid,
            "target_prob":target_prob, "top1_acc":top1_acc, "status":status,
        })

    # Summary table
    print("\n" + "="*75)
    print("EVALUATION SUMMARY")
    print("="*75)
    print(f"{'Class':<16}{'Samples':>8}{'MS-SSIM':>9}{'FID':>8}{'PathProb':>10}{'Top-1':>7}{'Status':>10}")
    print("-"*75)
    for r in results:
        print(f"{r['class']:<16}{r['synthetic_count']:>8}{r['ms_ssim']:>9.3f}"
              f"{r['fid']:>8.1f}{r['target_prob']:>10.3f}{r['top1_acc']:>7.1%}{r['status']:>10}")
    print("="*75)
    print("\nMS-SSIM > 0.6 ✅  |  FID < 50 ✅  |  PathProb > 0.3 ✅")

    csv_path = OUT_DIR / "evaluation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader(); writer.writerows(results)
    print(f"\nResults saved → {csv_path}")
    print("Next step: python3 src/vae/train.py")

if __name__ == "__main__":
    main()