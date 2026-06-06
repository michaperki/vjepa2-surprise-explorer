import gc
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _rss_gb() -> float:
    """Current resident set size in GiB (Linux /proc), 0.0 if unavailable."""
    try:
        with open(f"/proc/{os.getpid()}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.predictor import vit_predictor
from src.models.vision_transformer import vit_large_rope


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
FRAMES_PER_CLIP = 16
CONTEXT_FRAMES = 8
TARGET_FRAMES = 8
TUBELET_SIZE = 2
IMG_SIZE = 256
PATCH_SIZE = 16
GRID_T = FRAMES_PER_CLIP // TUBELET_SIZE
GRID_H = IMG_SIZE // PATCH_SIZE
GRID_W = IMG_SIZE // PATCH_SIZE
CONTEXT_TOKENS_T = GRID_T // 2
LOSS_EXP = 1.0


def build_video_transform():
    short_side_size = int(256.0 / 224 * IMG_SIZE)
    return video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(IMG_SIZE, IMG_SIZE)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


def clean_state_dict(state_dict):
    return {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state_dict.items()
    }


def build_encoder():
    return vit_large_rope(
        img_size=(IMG_SIZE, IMG_SIZE),
        patch_size=PATCH_SIZE,
        num_frames=FRAMES_PER_CLIP,
        tubelet_size=TUBELET_SIZE,
        uniform_power=True,
        use_sdpa=True,
    )


def build_predictor():
    return vit_predictor(
        img_size=(IMG_SIZE, IMG_SIZE),
        patch_size=PATCH_SIZE,
        num_frames=FRAMES_PER_CLIP,
        tubelet_size=TUBELET_SIZE,
        embed_dim=1024,
        predictor_embed_dim=384,
        depth=12,
        num_heads=12,
        num_mask_tokens=10,
        use_mask_tokens=True,
        use_rope=True,
        uniform_power=True,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
    )


def _as_tchw_tensor(clip):
    if isinstance(clip, np.ndarray):
        tensor = torch.from_numpy(clip)
    elif isinstance(clip, torch.Tensor):
        tensor = clip.detach().cpu()
    else:
        raise TypeError("clip must be a numpy array or torch tensor")

    if tensor.ndim != 4:
        raise ValueError(f"clip must have 4 dimensions, got {tuple(tensor.shape)}")

    if tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 3, 1, 2)
    elif tensor.shape[1] != 3:
        raise ValueError("clip must be shaped T,H,W,3 or T,3,H,W")

    return tensor.contiguous()


def _make_masks(device):
    tokens_per_t = GRID_H * GRID_W
    context_indices = torch.arange(0, CONTEXT_TOKENS_T * tokens_per_t, device=device)
    target_indices = torch.arange(CONTEXT_TOKENS_T * tokens_per_t, GRID_T * tokens_per_t, device=device)
    return context_indices.unsqueeze(0), target_indices.unsqueeze(0)


class SurpriseEngine:
    def __init__(self, checkpoint_path="checkpoints/vitl.pt", device=None, weights_dtype=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Optionally hold model weights in a lower precision (e.g. torch.bfloat16).
        # Compute already runs in bf16 via autocast, so this barely changes the
        # numbers but roughly halves resident GPU memory (~2.9GB -> ~1.5GB for
        # three ViT-L), which keeps a 6GB card from OOMing mid-run.
        self.weights_dtype = weights_dtype
        self.transform = build_video_transform()
        self.masks_enc, self.masks_pred = _make_masks(self.device)
        self.encoder, self.target_encoder, self.predictor = self._load_models(Path(checkpoint_path))
        if self.device.type == "cuda":
            self.autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            self.autocast_dtype = None
            self.precision = "fp32 (CPU fallback; no CUDA device visible)"
        if self.weights_dtype is not None:
            self.precision += f" (weights={self.weights_dtype})"

    def _load_models(self, checkpoint_path):
        # mmap=True keeps the multi-GB checkpoint memory-mapped instead of
        # reading it all into anonymous RAM, and we build/move one model at a
        # time so we never hold three fp32 ViT-L copies at once. Without this,
        # peak host RAM exceeds a small (~8GB) machine and the load is OOM-killed.
        print(f"[load] start  RSS={_rss_gb():.2f}GiB", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
        print(f"[load] mmap'd checkpoint  RSS={_rss_gb():.2f}GiB", flush=True)

        def load_one(builder, key, strict):
            model = builder()
            print(f"[load]   built {key}  RSS={_rss_gb():.2f}GiB", flush=True)
            sd = clean_state_dict(checkpoint[key])
            model.load_state_dict(sd, strict=strict)
            del sd
            model.to(self.device)
            if self.weights_dtype is not None:
                model.to(self.weights_dtype)
            model.eval()
            gc.collect()
            print(f"[load]   moved {key} to {self.device}  RSS={_rss_gb():.2f}GiB", flush=True)
            return model

        encoder = load_one(build_encoder, "encoder", False)
        target_encoder = load_one(build_encoder, "target_encoder", False)
        predictor = load_one(build_predictor, "predictor", True)
        del checkpoint
        gc.collect()
        print(f"[load] done  RSS={_rss_gb():.2f}GiB", flush=True)
        return encoder, target_encoder, predictor

    def _autocast_context(self):
        if self.device.type != "cuda":
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def _prepare_clips(self, context_clip, target_clip=None):
        context = _as_tchw_tensor(context_clip)
        if context.shape[0] != CONTEXT_FRAMES:
            raise ValueError(f"context_clip must contain {CONTEXT_FRAMES} frames")
        if target_clip is None:
            return context, None
        target = _as_tchw_tensor(target_clip)
        if target.shape[0] != TARGET_FRAMES:
            raise ValueError(f"target_clip must contain {TARGET_FRAMES} frames")
        return context, target

    def _context_input(self, context):
        target_zeros = torch.zeros(
            (TARGET_FRAMES, context.shape[1], context.shape[2], context.shape[3]),
            dtype=context.dtype,
        )
        context_only = torch.cat([context, target_zeros], dim=0)
        return self.transform(context_only).unsqueeze(0).to(self.device)

    def _actual_input(self, context, target):
        actual_future = torch.cat([context, target], dim=0)
        return self.transform(actual_future).unsqueeze(0).to(self.device)

    def _assemble_inputs(self, context_clip, target_clip):
        context, target = self._prepare_clips(context_clip, target_clip)
        target_zeros = torch.zeros_like(target)
        context_only = torch.cat([context, target_zeros], dim=0)
        actual_future = torch.cat([context, target], dim=0)
        context_x = self.transform(context_only).unsqueeze(0).to(self.device)
        actual_x = self.transform(actual_future).unsqueeze(0).to(self.device)
        return context_x, actual_x

    def compute_surprise(self, context_clip, target_clip):
        context_x, actual_x = self._assemble_inputs(context_clip, target_clip)
        with torch.inference_mode(), self._autocast_context():
            z_context = self.encoder(context_x, self.masks_enc)
            z_pred = self.predictor(z_context, self.masks_enc, self.masks_pred)
            h = self.target_encoder(actual_x)
            h = F.layer_norm(h, (h.size(-1),))
            h_target = torch.gather(
                h,
                dim=1,
                index=self.masks_pred.unsqueeze(-1).repeat(1, 1, h.size(-1)),
            )
            token_surprise = torch.mean(torch.abs(z_pred.float() - h_target.float()) ** LOSS_EXP, dim=-1) / LOSS_EXP
            scalar = token_surprise.mean()

        return scalar.item(), token_surprise.squeeze(0).cpu().numpy()

    def compute_surprises(self, context_clip, target_clips):
        context, _ = self._prepare_clips(context_clip)
        targets = [self._prepare_clips(context, target_clip)[1] for target_clip in target_clips]
        context_x = self._context_input(context)

        outputs = []
        with torch.inference_mode(), self._autocast_context():
            z_context = self.encoder(context_x, self.masks_enc)
            z_pred = self.predictor(z_context, self.masks_enc, self.masks_pred)
            for target in targets:
                actual_x = self._actual_input(context, target)
                h = self.target_encoder(actual_x)
                h = F.layer_norm(h, (h.size(-1),))
                h_target = torch.gather(
                    h,
                    dim=1,
                    index=self.masks_pred.unsqueeze(-1).repeat(1, 1, h.size(-1)),
                )
                token_surprise = (
                    torch.mean(torch.abs(z_pred.float() - h_target.float()) ** LOSS_EXP, dim=-1) / LOSS_EXP
                )
                outputs.append((token_surprise.mean().item(), token_surprise.squeeze(0).cpu().numpy()))

        return outputs

    def compute_surprise_split(self, window_clip, context_frames):
        """Surprise for a 16-frame window with a *configurable* context length.

        Replicates the variable-context protocol of Garrido et al. (2025,
        arXiv:2502.11831): encode the first ``context_frames`` and predict the
        representations of the remaining ``FRAMES_PER_CLIP - context_frames``
        future frames, scoring the same layer-normed mean-abs error. Their
        headline IntPhys numbers come from *per-property-optimal* context, often
        small (C=2 -> predict 14); the rest of this codebase only ever ran the
        fixed 8/8 split, so this method exists to sweep that axis.

        ``context_frames`` must be a positive multiple of ``TUBELET_SIZE`` and
        strictly less than ``FRAMES_PER_CLIP``.
        """
        if context_frames % TUBELET_SIZE or not (0 < context_frames < FRAMES_PER_CLIP):
            raise ValueError(
                f"context_frames must be a multiple of {TUBELET_SIZE} in (0, {FRAMES_PER_CLIP})"
            )
        window = _as_tchw_tensor(window_clip)
        if window.shape[0] != FRAMES_PER_CLIP:
            raise ValueError(f"window_clip must contain {FRAMES_PER_CLIP} frames")

        tokens_per_t = GRID_H * GRID_W
        context_tokens_t = context_frames // TUBELET_SIZE
        split = context_tokens_t * tokens_per_t
        masks_enc = torch.arange(0, split, device=self.device).unsqueeze(0)
        masks_pred = torch.arange(split, GRID_T * tokens_per_t, device=self.device).unsqueeze(0)

        # Context-only input: the real context frames, zero-padded where the
        # future will be predicted (those positions are masked out of the
        # encoder anyway), exactly as _context_input does for the 8/8 split.
        zeros = torch.zeros(
            (FRAMES_PER_CLIP - context_frames, window.shape[1], window.shape[2], window.shape[3]),
            dtype=window.dtype,
        )
        context_x = self.transform(torch.cat([window[:context_frames], zeros], dim=0)).unsqueeze(0).to(self.device)
        actual_x = self.transform(window).unsqueeze(0).to(self.device)

        with torch.inference_mode(), self._autocast_context():
            z_context = self.encoder(context_x, masks_enc)
            z_pred = self.predictor(z_context, masks_enc, masks_pred)
            h = self.target_encoder(actual_x)
            h = F.layer_norm(h, (h.size(-1),))
            h_target = torch.gather(h, dim=1, index=masks_pred.unsqueeze(-1).repeat(1, 1, h.size(-1)))
            token_surprise = torch.mean(torch.abs(z_pred.float() - h_target.float()) ** LOSS_EXP, dim=-1) / LOSS_EXP
            scalar = token_surprise.mean()

        return scalar.item(), token_surprise.squeeze(0).cpu().numpy()

    def embed_clip(self, context_clip, target_clip):
        """Mean-pooled target-encoder embedding of the full 16-frame clip.

        This is the representation the viewer's latent-space map projects:
        *where* a clip lands in V-JEPA's embedding space, independent of the
        prediction error. Comparing the embedding of a real vs. a scrambled
        future is the brittleness signal the robustness view visualizes.
        """
        context, target = self._prepare_clips(context_clip, target_clip)
        actual_x = self._actual_input(context, target)
        with torch.inference_mode(), self._autocast_context():
            h = self.target_encoder(actual_x)
            h = F.layer_norm(h, (h.size(-1),))
            pooled = h.float().mean(dim=1)
        return pooled.squeeze(0).cpu().numpy()


_DEFAULT_ENGINE = None


def compute_surprise(context_clip, target_clip, checkpoint_path="checkpoints/vitl.pt", device=None):
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = SurpriseEngine(checkpoint_path=checkpoint_path, device=device)
    return _DEFAULT_ENGINE.compute_surprise(context_clip, target_clip)
