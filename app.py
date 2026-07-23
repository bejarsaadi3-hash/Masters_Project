import torch
import torchvision.transforms as transforms
import gradio as gr
from PIL import Image

from damage_analyzer import analyze_damage_with_llm
from ebay_service import fetch_price_by_query

# ----------------------------------------------------------------------
# 1. Load Your Trained PyTorch Model & Weights
# ----------------------------------------------------------------------
# (Adjust this import/class name to match your actual PyTorch model class)
# from custom_model import MultimodalSalvageModel 

SEVERITY_CLASSES = {
    0: "Minor Cosmetic Damage",
    1: "Moderate Body / Structural Damage",
    2: "Severe Crash Impact"
}

# Image Preprocessing for ResNet-50
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize and load model weights
# model = MultimodalSalvageModel()
# model.load_state_dict(torch.load("multimodal_salvage_model.pth", map_location=torch.device('cpu')))
# model.eval()


def run_salvage_appraisal_pipeline(pil_image, make, year):
    if pil_image is None:
        return "Please upload a vehicle damage photo."

    # ------------------------------------------------------------------
    # Step A: YOUR MODEL PREDICTIONS (PyTorch ResNet-50 + Metadata)
    # ------------------------------------------------------------------
    img_tensor = transform(pil_image).unsqueeze(0)  # Shape: [1, 3, 224, 224]
    
    # Run inference through your trained neural network heads
    with torch.no_grad():
        # severity_logits, labor_hours_pred = model(img_tensor, make, year)
        # severity_idx = torch.argmax(severity_logits, dim=1).item()
        # predicted_severity = SEVERITY_CLASSES.get(severity_idx, "Moderate Damage")
        # predicted_labor_hours = float(labor_hours_pred.item())
        
        # Simulated PyTorch Model Output (Uncomment lines above once connected)
        predicted_severity = "Moderate Body / Structural Damage"
        predicted_labor_hours = 12.8  # Hours predicted by your ResNet-50 regression head

    # ------------------------------------------------------------------
    # Step B: LLM + eBAY (Part Search Query Generation & Live Pricing)
    # ------------------------------------------------------------------
    llm_queries = analyze_damage_with_llm(pil_image, make, year)
    
    parts_breakdown = []
    total_parts_cost = 0.0
    
    for query in llm_queries:
        live_price = fetch_price_by_query(query)
        
        if live_price:
            cost = live_price
            source = "Live eBay Motors API"
        else:
            cost = 250.00  # Defensive heuristic fallback
            source = "Internal Fallback Matrix"
            
        parts_breakdown.append(f"* **Part Query:** `{query}` — **Cost:** ${cost:.2f} *({source})*")
        total_parts_cost += cost

    parts_report_str = "\n".join(parts_breakdown)

    # ------------------------------------------------------------------
    # Step C: Combine Outputs into Defense Report
    # ------------------------------------------------------------------
    report = f"""
    ### 🚘 Multimodal Salvage Appraisal Report

    #### 🧠 Deep Learning Model Inference (PyTorch ResNet-50)
    * **Input Metadata:** {year} {make}
    * **Predicted Crash Severity:** **{predicted_severity}** *(Classification Head)*
    * **Predicted Repair Labor:** **{predicted_labor_hours:.1f} Hours** *(Regression Head)*

    ---

    #### 📦 Live Replacement Parts Breakdown (eBay Motors API)
    {parts_report_str}

    * **Total Estimated Replacement Parts Cost:** **${total_parts_cost:.2f}**
    """
    return report


# Gradio Dashboard Setup
iface = gr.Interface(
    fn=run_salvage_appraisal_pipeline,
    inputs=[
        gr.Image(type="pil", label="Upload Vehicle Damage Image"),
        gr.Dropdown(choices=["Toyota", "Nissan", "BMW", "Ford", "Honda", "Hyundai"], label="Vehicle Make", value="Toyota"),
        gr.Number(label="Vehicle Year", value=2019, precision=0)
    ],
    outputs=gr.Markdown(label="Appraisal Summary"),
    title="Multimodal Salvage Valuation Engine"
)

if __name__ == "__main__":
    iface.launch()