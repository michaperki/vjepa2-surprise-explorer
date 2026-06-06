import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.attentive_pooler import AttentiveClassifier
from src.models.vision_transformer import vit_large_rope


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def build_video_transform(img_size: int):
    short_side_size = int(256.0 / 224 * img_size)
    return video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


def clean_state_dict(state_dict):
    return {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state_dict.items()
    }


def load_encoder(checkpoint_path: Path, device: torch.device, frames_per_clip: int):
    model = vit_large_rope(
        img_size=(256, 256),
        patch_size=16,
        num_frames=frames_per_clip,
        tubelet_size=2,
        uniform_power=True,
        use_sdpa=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = clean_state_dict(checkpoint["target_encoder"])
    msg = model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model, msg


def load_classifier(checkpoint_path: Path, device: torch.device):
    classifier = AttentiveClassifier(
        embed_dim=1024,
        num_heads=16,
        depth=4,
        num_classes=174,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = clean_state_dict(checkpoint["classifiers"][0])
    msg = classifier.load_state_dict(state_dict, strict=True)
    classifier.to(device).eval()
    return classifier, msg


def load_clip(video_path: Path, frames_per_clip: int):
    vr = VideoReader(str(video_path))
    frame_idx = np.arange(frames_per_clip) * 4
    if frame_idx[-1] >= len(vr):
        frame_idx = np.linspace(0, len(vr) - 1, frames_per_clip).astype(np.int64)
    video = vr.get_batch(frame_idx).asnumpy()
    tensor = torch.from_numpy(video).permute(0, 3, 1, 2)
    return tensor, frame_idx, len(vr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/sample_video.mp4")
    parser.add_argument("--encoder", default="checkpoints/vitl.pt")
    parser.add_argument("--probe", default="checkpoints/ssv2-vitl-16x2x3.pt")
    parser.add_argument("--labels", default="data/ssv2_classes.json")
    parser.add_argument("--frames", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    labels = json.loads(Path(args.labels).read_text())
    transform = build_video_transform(img_size=256)

    clip, frame_idx, total_frames = load_clip(Path(args.video), args.frames)
    x = transform(clip).unsqueeze(0).to(device)

    print(f"Device: {device}")
    print(f"Video: {args.video}")
    print(f"Frames sampled: {frame_idx.tolist()} from {total_frames} total frames")
    print(f"Input tensor: {tuple(x.shape)}")

    encoder, encoder_msg = load_encoder(Path(args.encoder), device, args.frames)
    classifier, classifier_msg = load_classifier(Path(args.probe), device)
    print(f"Encoder load: {encoder_msg}")
    print(f"Probe load: {classifier_msg}")

    features = encoder(x)
    logits = classifier(features)
    probs = F.softmax(logits, dim=-1)

    print(f"V-JEPA 2 feature tensor: {tuple(features.shape)}")
    print(f"SSv2 probe logits: {tuple(logits.shape)}")
    print("Top 5 Something-Something-v2 predictions:")
    for rank, idx in enumerate(probs[0].topk(5).indices.tolist(), start=1):
        print(f"{rank}. {labels[str(idx)]}: {probs[0, idx].item() * 100:.2f}%")


if __name__ == "__main__":
    main()
