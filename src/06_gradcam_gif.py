"""
Step 6: Generate animated GradCAM GIFs for representative Normal / Abnormal videos.

Runs Grad-CAM on every optical flow frame of each representative video
and saves the overlays as an animated GIF.

Output:
    output/plots/gradcam_Normal.gif
    output/plots/gradcam_Abnormal.gif
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
from torchvision import models, transforms
from pathlib import Path
from PIL import Image

_BASE      = Path(__file__).resolve().parent.parent
MODEL_PATH = _BASE / "output" / "models" / "cnn_best.pth"
FLOW_DIR   = _BASE / "output" / "optical_flow"
PLOT_DIR   = _BASE / "output" / "plots"
IMG_SIZE   = 224
CLASSES    = ["Abnormal", "Normal"]
FPS        = 10   # GIF frame rate

REPR_VIDEOS = {
    "Normal":   "S01_Norm_S_01",
    "Abnormal": "A01_Act_S_01",
}

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output):
        self.activations = output.detach()

    def _save_gradient(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_cam(img_path: Path, cam: np.ndarray) -> np.ndarray:
    img     = cv2.imread(str(img_path))
    img     = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)


def make_gif(class_name: str, video_name: str, model, cam_gen, device) -> Path:
    video_dir   = FLOW_DIR / class_name / video_name
    frame_paths = sorted(video_dir.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found: {video_dir}")

    gif_frames = []
    class_idx  = CLASSES.index(class_name)

    for i, img_path in enumerate(frame_paths):
        img    = Image.open(img_path).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(device)
        tensor.requires_grad_(True)

        cam     = cam_gen.generate(tensor, class_idx)
        overlay = overlay_cam(img_path, cam)
        rgb     = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        gif_frames.append(Image.fromarray(rgb))

        if (i + 1) % 10 == 0:
            print(f"  [{class_name}] {i+1}/{len(frame_paths)} frames done")

    out_path = PLOT_DIR / f"gradcam_{class_name}.gif"
    gif_frames[0].save(
        out_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=1000 // FPS,
        loop=0,
    )
    return out_path


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    cam_gen = GradCAM(model, model.layer4[-1])

    for class_name, video_name in REPR_VIDEOS.items():
        print(f"\n[{class_name}] {video_name}")
        out = make_gif(class_name, video_name, model, cam_gen, device)
        print(f"  Saved: {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
