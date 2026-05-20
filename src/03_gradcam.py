"""
Step 3: Generate Grad-CAM visualizations for model interpretation.

Input : output/optical_flow/ + output/models/cnn_best.pth
Output: output/results/gradcam/
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import models, transforms
from pathlib import Path
from PIL import Image

MODEL_PATH  = Path("../output/models/cnn_best.pth")
INPUT_DIR   = Path("../output/optical_flow")
OUTPUT_DIR  = Path("../output/results/gradcam")
IMG_SIZE    = 224
CLASSES     = ["Abnormal", "Normal"]  # matches ImageFolder alphabetical order

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output):
        self.activations = output.detach()

    def _save_gradient(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_cam(image_path: Path, cam: np.ndarray) -> np.ndarray:
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)


def run_gradcam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=None)
    import torch.nn as nn
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    cam_gen = GradCAM(model, model.layer4[-1])

    for class_name in CLASSES:
        img_paths = list((INPUT_DIR / class_name).rglob("*.png"))
        out_dir = OUTPUT_DIR / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{class_name}] Generating Grad-CAM for {len(img_paths)} images")

        for img_path in img_paths:
            img = Image.open(img_path).convert("RGB")
            tensor = TRANSFORM(img).unsqueeze(0).to(device)
            pred_idx = model(tensor).argmax(1).item()
            cam = cam_gen.generate(tensor, pred_idx)
            overlay = overlay_cam(img_path, cam)
            cv2.imwrite(str(out_dir / img_path.name), overlay)

    print("Grad-CAM complete. Results saved to", OUTPUT_DIR)


if __name__ == "__main__":
    run_gradcam()
