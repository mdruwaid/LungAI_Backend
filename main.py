import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import base64
from typing import Optional
from PIL import Image
import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

app = FastAPI(title="Lung Cancer AI Clinical Decision Support API")

# ==========================================
# 1. LOAD THE AI MODELS
# ==========================================
print("Loading AI Models into memory...")

clinical_model = xgb.XGBClassifier()
clinical_model.load_model("xgboost_clinical_c2.json")
print("✅ XGBoost Model C2 Loaded!")

device = torch.device("cpu")
image_model = models.mobilenet_v2(weights=None)
image_model.classifier[1] = nn.Linear(image_model.last_channel, 3)
image_model.load_state_dict(torch.load("mobilenet_v2_lung_balanced.pth", map_location=device))
image_model.eval()
print("✅ PyTorch MobileNetV2 Loaded!")

classes = ["Normal", "Malignant", "Benign"]

# ==========================================
# 2. GRAD-CAM CLASS (Brought over from Notebook!)
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        model_output = self.model(input_tensor)
        self.model.zero_grad()
        target = model_output[0][class_idx]
        target.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        if torch.max(heatmap) != 0:
            heatmap /= torch.max(heatmap)
        return heatmap.cpu().numpy()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def process_clinical_data(age, gender, cigsmok, scr_res0, nodule_size, perp_dia, nodule_4mm, spiculated, ground_glass, growth):
    # Model C2 expects 47 features. We fill the most important ones based on your Feature Importance chart!
    features = np.zeros((1, 47), dtype=np.float32)
    if age is not None: features[0, 0] = age
    if gender is not None: features[0, 1] = gender
    if cigsmok is not None: features[0, 3] = cigsmok  # cigsmok
    if scr_res0 is not None: features[0, 4] = scr_res0 # screening result
    
    # Direct CT baseline features
    if nodule_size is not None: features[0, 6] = nodule_size
    if perp_dia is not None: features[0, 7] = perp_dia
    if nodule_4mm is not None: features[0, 9] = nodule_4mm
    if spiculated is not None: features[0, 11] = spiculated
    if ground_glass is not None: features[0, 12] = ground_glass
    
    # Longitudinal feature
    if growth is not None: features[0, 14] = growth 
    
    # Predict Early Cancer Probability
    prob = clinical_model.predict_proba(features)[0, 1]
    return f"{prob * 100:.1f}%"

async def process_image_data(file: UploadFile):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 1. Preprocess for Model
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # 2. Get Prediction
    with torch.no_grad():
        outputs = image_model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        class_idx = torch.argmax(probabilities).item()
        
    prediction = classes[class_idx]
    confidence = probabilities[class_idx].item()

    # 3. Generate Grad-CAM Heatmap
    target_layer = image_model.features[-1]
    grad_cam = GradCAM(image_model, target_layer)
    
    # We have to re-enable gradients temporarily for Grad-CAM
    with torch.enable_grad():
        img_tensor.requires_grad_()
        heatmap = grad_cam.generate_heatmap(img_tensor, class_idx)
    
    # 4. Create the Overlay (Fixing the pink warning!)
    heatmap_resized = cv2.resize(heatmap, (img_resized.shape[1], img_resized.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = np.float32(heatmap_colored) / 255
    heatmap_colored = heatmap_colored[..., ::-1] # BGR to RGB
    
    original_img_norm = img_resized.astype(np.float32) / 255.0
    superimposed_img = heatmap_colored * 0.4 + original_img_norm
    superimposed_img = np.clip(superimposed_img, 0, 1) # Clamped perfectly
    
    # 5. Convert Image to Base64 String so the Mobile App can display it
    superimposed_img_uint8 = (superimposed_img * 255).astype(np.uint8)
    pil_img = Image.fromarray(superimposed_img_uint8)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return prediction, f"{confidence * 100:.1f}%", img_base64

# ==========================================
# 4. DYNAMIC ROUTING ENDPOINT
# ==========================================
@app.post("/analyze")
async def analyze_patient(
    # Expanded Professional Clinical Inputs
    age: Optional[int] = Form(None),
    gender: Optional[int] = Form(None),
    cigsmok: Optional[int] = Form(None),
    scr_res0: Optional[int] = Form(None, description="Screening Result Code"),
    nodule_size: Optional[float] = Form(None, description="Max Nodule Size (mm)"),
    perp_dia: Optional[float] = Form(None, description="Perpendicular Diameter (mm)"),
    nodule_4mm: Optional[int] = Form(None, description="Count of Nodules >= 4mm"),
    spiculated: Optional[int] = Form(None, description="Spiculated Margins (0=No, 1=Yes)"),
    ground_glass: Optional[int] = Form(None, description="Ground Glass Opacity (0=No, 1=Yes)"),
    growth: Optional[int] = Form(None, description="Longitudinal Growth Detected (0=No, 1=Yes)"),
    
    # Image Input
    ct_image: Optional[UploadFile] = File(None)
):
    has_clinical = any(v is not None for v in [age, gender, cigsmok, nodule_size, growth])
    has_image = ct_image is not None

    if has_clinical and has_image:
        risk_score = process_clinical_data(age, gender, cigsmok, scr_res0, nodule_size, perp_dia, nodule_4mm, spiculated, ground_glass, growth)
        img_pred, img_conf, heatmap_b64 = await process_image_data(ct_image)
        return {
            "scenario": "Decoupled - Both Models Triggered",
            "clinical_early_cancer_risk": risk_score,
            "image_prediction": img_pred,
            "image_confidence": img_conf,
            "heatmap_image_base64": heatmap_b64,
            "ai_report": f"Clinical features predict an Early Cancer Risk of {risk_score}. CNN imaging independently detects a {img_conf} confidence {img_pred} mass. See attached spatial heatmap."
        }

    elif has_image and not has_clinical:
        img_pred, img_conf, heatmap_b64 = await process_image_data(ct_image)
        return {
            "scenario": "Image Only Pipeline",
            "image_prediction": img_pred,
            "image_confidence": img_conf,
            "heatmap_image_base64": heatmap_b64,
            "ai_report": f"Clinical history unavailable. CNN imaging analysis detects a {img_conf} confidence {img_pred} anomaly. Refer to Grad-CAM heatmap."
        }

    elif has_clinical and not has_image:
        risk_score = process_clinical_data(age, gender, cigsmok, scr_res0, nodule_size, perp_dia, nodule_4mm, spiculated, ground_glass, growth)
        return {
            "scenario": "Clinical Only Pipeline",
            "clinical_early_cancer_risk": risk_score,
            "ai_report": f"No imaging provided. Patient factors indicate a {risk_score} probability of early-stage lung cancer."
        }
    else:
        return {"error": "Please provide clinical data or a CT image."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
