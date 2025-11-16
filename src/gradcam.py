import cv2
import torch
import numpy as np
from model import create_model


class GradCAM:
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, class_idx):
        gradients = self.gradients.to(self.device)
        activations = self.activations.to(self.device)

        weights = torch.mean(gradients, dim=(2, 3)).squeeze()

        cam = torch.zeros(activations.shape[2:], dtype=torch.float32).to(self.device)

        for i, w in enumerate(weights):
            cam += w * activations[0, i, :, :]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)

        return cam.detach().cpu().numpy()



def generate_gradcam(image_path, weights, model_name="efficientnet_b3", size=224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = create_model(model_name=model_name, num_classes=5, pretrained=False)
    ckpt = torch.load(weights, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # LOAD IMAGE
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image: " + image_path)

    orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(orig, (size, size))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # IMPORTANT: correct target layer for TIMM EfficientNet-B3
    target_layer = model.conv_head

    cam_obj = GradCAM(model, target_layer, device)

    # Forward pass
    scores = model(img_tensor)
    pred = torch.argmax(scores, dim=1).item()

    # Backward pass
    model.zero_grad()
    scores[0, pred].backward()

    heatmap = cam_obj.generate(pred)

    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(orig, 0.6, heatmap_colored, 0.4, 0)

    return pred, overlay
