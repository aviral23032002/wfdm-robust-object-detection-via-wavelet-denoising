"""
train_learnwfdm.py — LearnableWFDM P1 Wavelet Training
=======================================================
STRATEGY:
  - LearnableWFDM inserted at index 1 (P1, 320x320 — highest resolution)
  - Per-channel learnable soft threshold per sub-band (LH, HL, HH)
  - Preserves LL (structure/geometry) completely
  - Alpha residual gate (zero-init) — safe identity at start
  - Pretrained backbone weights transferred with +1 index shift
"""

import time
import torch
import ultralytics.nn.tasks as tasks

# Inject BEFORE YOLO import
from models.wfdm import LearnableWFDM, HaarDWT, HaarIWT
tasks.LearnableWFDM = LearnableWFDM


from ultralytics import YOLO


def sanity_check():
    """Verify Haar round-trip and soft threshold sanity."""
    print("=" * 65)
    print("STEP 1: SANITY CHECKS")
    print("=" * 65)

    # 1a. Round-trip check
    all_ok = True
    for c in [3, 16, 32]:
        dwt = HaarDWT(c); iwt = HaarIWT(c)
        x = torch.randn(2, c, 64, 64)
        rec = iwt(dwt(x))
        ok = torch.allclose(x, rec, atol=1e-5)
        all_ok = all_ok and ok
        print(f"  Haar round-trip ch={c:<3}: {'✅ PASS' if ok else '❌ FAIL'}")
    if not all_ok:
        raise RuntimeError("Haar round-trip failed — check wfdm.py")

    # 1b. Soft threshold sanity: noisy signal should come out cleaner
    lwfdm = LearnableWFDM(3)
    x_clean = torch.randn(1, 3, 64, 64) * 0.5
    x_noisy = x_clean + torch.randn_like(x_clean) * 0.3
    # Trigger lazy init first (alpha/dwt/iwt are created on first forward)
    with torch.no_grad():
        lwfdm(x_noisy)
        # Now force alpha=1 so denoising is fully applied for this test
        lwfdm.alpha.fill_(1.0)
        out = lwfdm(x_noisy)
    mse_noisy = (x_noisy - x_clean).pow(2).mean().item()
    mse_out   = (out - x_clean).pow(2).mean().item()
    improved  = mse_out < mse_noisy
    print(f"  Soft-threshold denoising test: {'✅ PASS' if improved else '⚠️  MARGINAL'} "
          f"(noisy MSE={mse_noisy:.4f} → denoised MSE={mse_out:.4f})")
    print(f"  Sanity checks complete.\n")


def load_pretrained_p1(model, pt_path: str):
    """
    Transfer weights from yolov8n.pt into our P1-shifted architecture.

    Index mapping:
      Pretrained layer 0   → New layer 0  (Conv P1/2, exact match)
      Pretrained layer k   → New layer k+1 (k=1..21, shift +1 for inserted LearnWFDM)
      Pretrained Detect    → SKIP (different nc: 80 vs 12)
    """
    print(f"[Step 3] Transferring weights from {pt_path}...")
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    src = ckpt['model'].float().state_dict() if isinstance(ckpt, dict) else ckpt.state_dict()

    tgt = model.model.state_dict()
    new_sd = {}
    matched = 0

    for key, val in src.items():
        if not key.startswith('model.'):
            new_key = key
        else:
            parts = key.split('.')
            idx = int(parts[1])
            # layer 0 → 0 (unchanged), layers 1+ → idx+1 (shifted past LearnableWFDM)
            if idx >= 1:
                parts[1] = str(idx + 1)
            new_key = '.'.join(parts)

        if new_key in tgt and val.shape == tgt[new_key].shape:
            new_sd[new_key] = val
            matched += 1

    model.model.load_state_dict(new_sd, strict=False)
    nc = model.model.yaml['nc']
    print(f"  ✅ Mapped {matched} pretrained layers.")
    print(f"  ✅ Detection head randomly initialized for {nc} classes.\n")
    return model


def main():
    sanity_check()

    # ------------------------------------------------------------------
    # 2. Build architecture
    # ------------------------------------------------------------------
    print("=" * 65)
    print("STEP 2: BUILDING LearnableWFDM P1 ARCHITECTURE")
    print("=" * 65)
    model = YOLO('yolov8n-learnwfdm.yaml', task='detect')
    nc = model.model.yaml['nc']
    print(f"  Architecture built. nc={nc}")

    # Spot-check the wavelet module is in the right place
    layer1 = model.model.model[1]
    assert isinstance(layer1, LearnableWFDM), \
        f"Expected LearnableWFDM at index 1, got {type(layer1)}"
    print(f"  ✅ LearnableWFDM confirmed at backbone index 1 (P1, highest resolution).\n")

    # ------------------------------------------------------------------
    # 3. Transfer pretrained backbone weights
    # ------------------------------------------------------------------
    model = load_pretrained_p1(model, 'yolov8n.pt')

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    print("=" * 65)
    print("STEP 4: TRAINING")
    print("=" * 65)
    start = time.time()

    model.train(
        data='data/exdark_corrupt_all.yaml',  # val=corrupt → best.pt selected for corruption
        epochs=100,
        imgsz=640,
        device='mps',
        batch=16,
        project='wfdm_runs',
        name='learnwfdm_p1_corrval',  # v2: val on corrupt
        plots=True,
        verbose=True,
        amp=True,
        cache=True,
        lr0=0.01,
        warmup_epochs=5.0,
        cos_lr=True,
        close_mosaic=10,
    )

    elapsed = time.time() - start
    h, rem = divmod(elapsed, 3600)
    m, s   = divmod(rem, 60)
    print(f"\n✅ Training complete in {int(h)}h {int(m)}m {s:.1f}s")

    # ------------------------------------------------------------------
    # 5. Evaluate on corrupted test set
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 5: EVALUATION ON CORRUPTED TEST SET")
    print("=" * 65)
    metrics = model.val(data='data/exdark_corrupt_test.yaml', split='test')
    print(f"\nFINAL RESULTS:")
    print(f"  mAP@0.5   : {metrics.box.map50:.4f}")
    print(f"  Precision : {metrics.box.p.mean():.4f}")
    print(f"  Recall    : {metrics.box.r.mean():.4f}")
    print(f"\n  Baseline to beat: 0.4400 (corrupt-trained YOLOv8n)")


if __name__ == '__main__':
    main()
