"""
train_simplewfdm.py — SimpleWFDM P2 Training Script
====================================================
KEY FIX vs all previous attempts:
  1. NO alpha gate → thresholds get real gradients from epoch 1
  2. Placed at P2 (32ch, 160x160) → 2x more capacity than P1
  3. val = val_corrupt → best.pt selected for corruption performance
  4. IWT(DWT(x)) = x at thresh=0 → mathematically safe identity
"""

import time
import torch
import ultralytics.nn.tasks as tasks

from models.wfdm import SimpleWFDM, HaarDWT, HaarIWT
tasks.SimpleWFDM = SimpleWFDM

from ultralytics import YOLO


def sanity_check():
    print("=" * 65)
    print("STEP 1: SANITY CHECKS")
    print("=" * 65)

    # Haar round-trip
    for c in [3, 32, 64]:
        dwt = HaarDWT(c); iwt = HaarIWT(c)
        x = torch.randn(2, c, 64, 64)
        ok = torch.allclose(x, iwt(dwt(x)), atol=1e-5)
        print(f"  Haar round-trip ch={c:<3}: {'✅ PASS' if ok else '❌ FAIL'}")

    # thresh=0 identity check (proves no-gate approach is safe)
    s = SimpleWFDM()
    x = torch.randn(1, 32, 64, 64)
    with torch.no_grad():
        s(x)  # trigger lazy init
        out = s(x)
    is_identity = torch.allclose(x, out, atol=1e-5)
    print(f"  thresh=0 identity:       {'✅ PASS' if is_identity else '❌ FAIL'} "
          f"(max diff={( x - out).abs().max().item():.2e})")

    # Gradient check: thresholds should get non-zero gradient
    s2 = SimpleWFDM()
    x2 = torch.randn(1, 32, 64, 64, requires_grad=False)
    s2(x2)  # lazy init
    x2 = torch.randn(1, 32, 64, 64)
    out2 = s2(x2)
    out2.sum().backward()
    grad_hh = s2.thresh_hh.grad.abs().mean().item()
    print(f"  thresh_hh gradient:      {'✅ NON-ZERO' if grad_hh > 1e-6 else '❌ ZERO'} "
          f"(mean={grad_hh:.4f})")
    print()


def load_pretrained_p2(model, pt_path: str):
    """
    Transfer yolov8n.pt weights into P2-shifted architecture.

    Layer mapping (SimpleWFDM inserted at index 3):
      Pretrained 0,1,2 → New 0,1,2  (Conv, Conv, C2f — exact match)
      Pretrained k≥3   → New k+1    (shift past SimpleWFDM)
      Detect head (80 classes) → SKIP (different nc)
    """
    print(f"[Step 3] Transferring weights from {pt_path} ...")
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    src  = ckpt['model'].float().state_dict() if isinstance(ckpt, dict) else ckpt.state_dict()

    tgt = model.model.state_dict()
    new_sd, matched = {}, 0

    for key, val in src.items():
        if not key.startswith('model.'):
            new_key = key
        else:
            parts = key.split('.')
            idx = int(parts[1])
            if idx >= 3:              # shift layers after SimpleWFDM insertion
                parts[1] = str(idx + 1)
            new_key = '.'.join(parts)

        if new_key in tgt and val.shape == tgt[new_key].shape:
            new_sd[new_key] = val
            matched += 1

    model.model.load_state_dict(new_sd, strict=False)
    print(f"  ✅ Matched {matched} pretrained layers.")
    print(f"  ✅ Detection head initialized for {model.model.yaml['nc']} classes.\n")
    return model


def main():
    sanity_check()

    # 2. Build model
    print("=" * 65)
    print("STEP 2: BUILDING SimpleWFDM P2 ARCHITECTURE (YOLOv8s)")
    print("=" * 65)
    model = YOLO('yolov8s-simplewfdm-p2.yaml', task='detect')
    layer3 = model.model.model[3]
    assert isinstance(layer3, SimpleWFDM), \
        f"Expected SimpleWFDM at index 3, got {type(layer3)}"
    print(f"  ✅ SimpleWFDM confirmed at backbone index 3 (P2, 160x160).")
    print(f"  nc = {model.model.yaml['nc']}\n")

    # 3. Transfer weights
    model = load_pretrained_p2(model, 'yolov8s.pt')

    # 4. Train
    print("=" * 65)
    print("STEP 4: TRAINING (val=corrupt → best.pt for corruption)")
    print("=" * 65)
    start = time.time()

    model.train(
        data='data/exdark_corrupt_all.yaml',   # val=val_corrupt ← KEY
        epochs=100,
        imgsz=640,
        device='mps',
        batch=16,
        project='wfdm_runs',
        name='simplewfdm_p2_v8s',
        plots=True,
        verbose=True,
        amp=True,
        cache=True,
        lr0=0.01,
        warmup_epochs=5.0,
        cos_lr=True,
        close_mosaic=10,
    )

    h, rem = divmod(time.time() - start, 3600)
    m, s = divmod(rem, 60)
    print(f"\n✅ Training complete in {int(h)}h {int(m)}m {s:.0f}s")

    # 5. Evaluate on held-out corrupt test set
    print("\n" + "=" * 65)
    print("STEP 5: FINAL EVALUATION ON CORRUPT TEST SET")
    print("=" * 65)
    for ckpt_name in ['best', 'last']:
        ckpt_path = f'wfdm_runs/simplewfdm_p2_nogate/weights/{ckpt_name}.pt'
        try:
            m = YOLO(ckpt_path)
            metrics = m.val(data='data/exdark_corrupt_test.yaml', split='test')
            print(f"  {ckpt_name}.pt → mAP@0.5={metrics.box.map50:.4f}  "
                  f"P={metrics.box.p.mean():.4f}  R={metrics.box.r.mean():.4f}")
        except Exception as e:
            print(f"  {ckpt_name}.pt → {e}")

    print(f"\n  Baseline to beat: 0.37 (corrupt-train YOLOv8n on corrupt test)")


if __name__ == '__main__':
    main()
