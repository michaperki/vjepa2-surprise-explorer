# Single-clip V-JEPA 2 run

This checkout was cloned from `https://github.com/facebookresearch/vjepa2`.

Assets downloaded:

- `checkpoints/vitl.pt`: official V-JEPA 2 ViT-L/16 checkpoint.
- `checkpoints/ssv2-vitl-16x2x3.pt`: official Something-Something-v2 attentive probe.
- `data/sample_video.mp4`: demo clip from the upstream `vjepa2_demo.py`.
- `data/ssv2_classes.json`: Something-Something-v2 label map.

The machine had no visible CUDA device, so the run used CPU.

Command:

```bash
PYTHONPATH=. python3 run_single_clip_vitl.py
```

Observed output:

```text
Device: cpu
Video: data/sample_video.mp4
Frames sampled: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60] from 150 total frames
Input tensor: (1, 3, 16, 256, 256)
Encoder load: <All keys matched successfully>
Probe load: <All keys matched successfully>
V-JEPA 2 feature tensor: (1, 2048, 1024)
SSv2 probe logits: (1, 174)
Top 5 Something-Something-v2 predictions:
1. Stuffing [something] into [something]: 48.88%
2. Putting [something] into [something]: 39.81%
3. Attaching [something] to [something]: 1.67%
4. Failing to put [something] into [something] because [something] does not fit: 1.41%
5. Closing [something]: 1.13%
```
