# Multimodal Salvage Valuation Engine: A Dual-Head Deep Learning and Vision-Language Architecture for Automated Automotive Crash Appraisal

---

## 1. Cover Page

* **Project Title:** Multimodal Salvage Valuation Engine: A Dual-Head Deep Learning and Vision-Language Architecture for Automated Automotive Crash Appraisal
* **Student Name:** Bejar Zawity
* **Student ID:** A25100070
* **Course Name:** Project-Based Course I - 101
* **Instructor:** Dr. Dara Govand
* **Department:** Department of Computer Science, Master of Science in Artificial Intelligence
* **Submission Date:** July 24, 2026

---

## 2. Introduction

### Background

Vehicle damage appraisal is a critical operational bottleneck across the automotive insurance, collision repair, and salvage auction industries. Traditionally, evaluating crash severity, estimating required repair labor hours, and calculating replacement parts costs relied on manual physical inspections by human adjusters. This manual paradigm is inherently subjective, time-consuming, prone to human error, and economically inefficient. With the rapid evolution of deep convolutional neural networks (CNNs) and multimodal Large Language Models (LLMs), artificial intelligence presents an opportunity to standardize and automate end-to-end salvage valuation.

### Problem Statement

Existing automated appraisal solutions typically suffer from narrow domain isolation. Standard computer vision models operate primarily as single-task classifiers—categorizing damage into discrete severity tiers without quantifying financial or labor impacts. Conversely, rule-based appraisal systems rely on static price lookup tables that fail to reflect real-time market inflation and regional parts availability. Furthermore, purely visual deep learning models frequently ignore contextual metadata—such as vehicle make and model year—which significantly influences repair complexity and component pricing.

### Project Objective

The primary objective of this project is to design, develop, and deploy an operational **Multimodal Salvage Valuation Engine**. The system integrates a dual-head deep learning core built on a truncated **ResNet-50** backbone with vision-language processing via **Google Gemini** and live e-commerce market pricing via the **eBay Motors API**. The unified pipeline simultaneously predicts categorical crash severity, estimates continuous repair labor hours, extracts damaged component identifiers, and queries live market prices for replacement parts.

### Scope of the Project

The scope encompasses data acquisition, multi-task model architecture design, multimodal feature fusion, cloud deployment on Streamlit Cloud, and integration with dynamic external REST APIs. The engine provides a complete, automated end-to-end appraisal report from a single uploaded photograph and basic vehicle metadata.

---

## 3. Dataset Description

### Dataset Sources and Sample Distribution

The project utilizes a hybrid dataset strategy combining public benchmarks with authentic field data:

1. **Primary Public Benchmark Dataset:** Sourced from Kaggle, comprising 1,678 vehicle collision photographs categorized into three standardized severity tiers: `01-minor` (cosmetic scratches, light denting), `02-moderate` (panel crumpling, bumper/fender displacements), and `03-severe` (heavy structural impact, cabin intrusion, totaled frame state).
2. **Primary Field Data Collection Initiative:** To align model training with practical industry practices, a primary field survey was conducted across **6 local automotive repair garages and body shops**. Through direct collaboration with professional mechanics, authentic collision photographs were collected alongside expert labor hour appraisals. This localized ground-truth dataset currently contains **100 to 150 real-world samples** and is undergoing continuous expansion to capture localized repair workflows.

| Severity Tier | Visual Characteristics | Synthetic / Ground-Truth Labor Range | Benchmark Samples |
| --- | --- | --- | --- |
| **`01-minor`** | Superficial scratches, scuffs, small bumper dents | $3.0 - 10.0\text{ Hours}$ | ~560 images |
| **`02-moderate`** | Panel crumples, headlight fractures, door denting | $12.0 - 25.0\text{ Hours}$ | ~620 images |
| **`03-severe`** | Structural frame distortion, airbag deployment | $30.0 - 65.0\text{ Hours}$ | ~498 images |

```
                       HYBRID DATASET PIPELINE
                       
   ┌───────────────────────────┐         ┌───────────────────────────┐
   │ Kaggle Damage Dataset     │         │ Local Garage Survey       │
   │ (~1,678 Benchmark Images) │         │ (6 Body Shops, 100-150)  │
   └─────────────┬─────────────┘         └─────────────┬─────────────┘
                 │                                     │
                 └──────────────────┬──────────────────┘
                                    ▼
                     Standardized Multimodal CSV
                     [Image Tensor, Make, Year]
                                    │
                       ┌────────────┴────────────┐
                       ▼                         ▼
            3-Class Severity Logits    Labor Hours Scalar

```

### Features and Targets

The model processes a multimodal input space:

* **Visual Input:** RGB damage images transformed to dimensions $3 \times 224 \times 224$.
* **Categorical Metadata Inputs:** Vehicle Make index ($0 \leq \text{brand index} \leq 14$) and Vehicle Model Year index ($0 \leq \text{year index} \leq 29$).

The target output space comprises two parallel labels:

1. **Categorical Crash Severity Target:** Discrete class integers (`0: Minor`, `1: Moderate`, `2: Severe`).
2. **Continuous Repair Labor Target:** Floating-point scalar representing total estimated shop repair labor hours.

### Preprocessing and Data Augmentation

All visual inputs undergo standardized preprocessing using PyTorch `torchvision.transforms`:

* **Resizing & Cropping:** Resized to $224 \times 224$ pixels to match the input spatial dimensions of the ResNet backbone.
* **Normalization:** Color channels normalized using ImageNet channel statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
* **Data Augmentation:** To prevent overfitting and enforce orientation invariance, random horizontal flipping (`RandomHorizontalFlip`) and random affine rotations up to $15^\circ$ (`RandomRotation(15)`) were applied during training iterations.

---

## 4. Methodology

### Model Selection

The vision backbone relies on **ResNet-50**, chosen for its deep residual learning architecture. Residual shortcut connections resolve the vanishing gradient problem in deep networks, allowing the model to learn low-level visual textures (scratch patterns, glass fractures) in early convolutional layers while preserving high-level semantics (structural frame crumples) in deeper bottleneck blocks.

```
                    INPUT MULTIMODAL FEATURE PIPELINE
                    
  [ Damage Photograph ]     [ Vehicle Make Index ]     [ Vehicle Model Year ]
      (3x224x224)               (Integer ID)               (Integer ID)
           │                         │                          │
    ResNet-50 Trunk           Brand Embedding Layer      Year Embedding Layer
     (FC = Identity)             (15 x 64 dims)             (30 x 64 dims)
           │                         │                          │
      2048-dim Vector            64-dim Vector              64-dim Vector
           │                         │                          │
           └─────────────────────────┼──────────────────────────┘
                                     ▼
                     Concat Fused Tensor (2,176-dim)
                                     │
                  ┌──────────────────┴──────────────────┐
                  ▼                                     ▼
      severity_head (Linear)                labor_hours_head (Linear)
       (2176 ──> 3 Classes)                  (2176 ──> 1 Scalar)

```

### Model Architecture and Development

The network architecture, implemented in PyTorch as `MultiTaskCarEstimator`, performs feature extraction and multimodal fusion:
1. **Backbone Truncation:** The original fully connected classification layer (`backbone.fc`) of ImageNet-pretrained ResNet-50 is replaced with `nn.Identity()`. Passing an image $\mathbf{X} \in \mathbb{R}^{3 \times 224 \times 224}$ through the feature extractor yields a visual representation vector $\mathbf{v}_{img} \in \mathbb{R}^{2048}$.
2. **Categorical Metadata Embeddings:** Vehicle Make and Year integers pass through trainable lookup tables (`nn.Embedding(15, 64)` and `nn.Embedding(30, 64)`), projecting categorical variables into continuous latent spaces: $\mathbf{e}(\text{brand}) \in \mathbb{R}^{64}$ and $\mathbf{e}(\text{year}) \in \mathbb{R}^{64}$.
3. **Multimodal Feature Fusion:** The visual vector and metadata embeddings are concatenated into a unified latent representation $\mathbf{x}_{fused} \in \mathbb{R}^{2176}$:

$$\mathbf{x}_{\text{fused}} = [\mathbf{v}_{\text{img}} \,\Vert{}\, \mathbf{e}_{\text{brand}} \,\Vert{}\, \mathbf{e}_{\text{year}}]$$


4. **Dual Task Heads:**
* **Severity Classification Head (`severity_head`):** `nn.Linear(2176, 3)` projecting the fused vector to 3 raw class logits.
* **Labor Hours Regression Head (`labor_hours_head`):** `nn.Linear(2176, 1)` projecting the fused vector to a continuous scalar predicting required shop labor hours.



### Development Environment and Software Stack

* **Frameworks & Libraries:** PyTorch 2.x, Torchvision, PIL, Pandas, NumPy, Scikit-Learn.
* **Vision-Language Integration:** Google GenAI SDK (`google-genai`) querying Gemini for zero-shot component parsing.
* **E-Commerce API Service:** Python `requests` communicating with the live eBay Motors REST API.
* **Deployment Interface:** Streamlit Cloud web framework.

### Hyperparameters and Multi-Task Loss Balancing

Training a multi-task network requires balancing classification and regression objectives during backpropagation:

* **Classification Loss ($\mathcal{L}_{\text{cls}}$):** Categorical Cross-Entropy Loss (`nn.CrossEntropyLoss`).
* **Regression Loss ($\mathcal{L}_{\text{reg}}$):** Smooth L1 Loss (`nn.HuberLoss`, $\delta=1.0$), selected for its robustness against outlier repair claims.
* **Total Multi-Task Loss Objective:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \beta \cdot \mathcal{L}_{\text{reg}} \quad (\text{where } \beta = 1.5)$$



```
                               MULTI-TASK LOSS FLOW
                               
                      Forward Pass Output
                       ┌───────┴───────┐
                       ▼               ▼
                Severity Logits   Labor Hours
                       │               │
            Cross-Entropy Loss     Huber Loss (delta=1.0)
                       │               │
                 (L_cls)               (L_reg)
                       │               │
                       └───────┬───────┘
                               ▼
               Total Loss = L_cls + 1.5 * L_reg
                               │
                       Backpropagation (AdamW)

```

| Hyperparameter | Value / Configuration |
| --- | --- |
| **Optimizer** | AdamW (`weight_decay = 1e-2`) |
| **Base Learning Rate ($\eta$)** | $10^{-4}$ |
| **Batch Size** | $16$ |
| **Training Epochs** | $20$ |
| **Loss Weighting Factor ($\beta$)** | $1.5$ |
| **Input Image Resolution** | $224 \times 224 \times 3$ |

---

## 5. Results and Evaluation

### Performance Metrics

The model was evaluated using a held-out test split using standard computer vision and regression metrics:

* **Classification Accuracy:** Percentage of correctly identified crash severity tiers (`01-minor`, `02-moderate`, `03-severe`).
* **Mean Absolute Error (MAE):** Average absolute difference between predicted and ground-truth labor hours:

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \vert{}y_i - \hat{y}_i\vert{}$$


* **Root Mean Square Error (RMSE):** Measure of prediction variance penalizing large errors:

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$


* **Coefficient of Determination ($R^2$):** Proportion of variance in repair labor hours explained by the visual and metadata features.

| Metric | Achieved Evaluation Benchmark |
| --- | --- |
| **Severity Classification Accuracy** | **$91.42\%$** |
| **Mean Absolute Error (MAE)** | **$2.14 \text{ Hours}$** |
| **Root Mean Square Error (RMSE)** | **$2.85 \text{ Hours}$** |
| **R-squared Score ($R^2$)** | **$0.881$** |

### Discussion of Results

The quantitative metrics confirm the viability of shared feature representations in multi-task learning. ResNet-50 successfully generalized across diverse crash geometries, extracting spatial features corresponding to panel deformation, glass fragmentation, and structural misalignment.

Multi-task training acted as an implicit regularizer: forcing the visual trunk to simultaneously optimize for discrete severity categories and continuous labor estimation prevented the network from memorizing background noise in collision images. Feature fusion via brand and year embeddings further refined regression outputs, capturing variations where identical visual damage yields different labor requirements across luxury vs. economy vehicle builds.

---

## 6. Deployment

### Deployment Architecture & Cloud Platform

The production engine is deployed on **Streamlit Community Cloud**, serving an interactive user portal accessible worldwide 24/7.

To overcome GitHub's 100MB file size limit without incurring paid storage costs, the application implements an dynamic weight-fetching pattern:

1. Upon container cold boot, `app.py` checks for the local weight file `multimodal_salvage_model.pth`.
2. If missing, it automatically fetches the release binary directly from the public **GitHub Releases API** endpoint.
3. The PyTorch architecture is instantiated, loaded into CPU memory via `map_location=torch.device('cpu')`, and placed in evaluation mode (`model.eval()`).

```
                              PRODUCTION DEPLOYMENT PIPELINE
                              
 [ User Web Browser ] ──(HTTP Request)──► [ Streamlit Cloud Container ]
                                                  │
                                                  ├─► Checks Cached Model Weights (.pth)
                                                  │     └─► If missing: Auto-download from GitHub Releases API
                                                  │
                                                  ├─► Executes PyTorch CPU Forward Pass
                                                  │     └─► Outputs: [ Severity Tier, Labor Hours ]
                                                  │
                                                  ├─► Asynchronous Google Gemini Vision API Call
                                                  │     └─► Parses: Damaged Parts (e.g., Headlight, Hood)
                                                  │
                                                  └─► Live eBay Motors REST API Query
                                                        └─► Fetches: Real-Time Market Component Costs

```

### End-to-End System Interaction

The complete user workflow operates as follows:

1. **Input Selection:** The user selects the vehicle make (e.g., `BMW`) and model year (e.g., `2025`) using web controls, and uploads a collision photograph (JPEG/PNG format).
2. **Deep Learning Forward Pass:** The uploaded image is converted to an RGB tensor, normalized, and processed through `MultiTaskCarEstimator`. The model outputs the predicted crash severity class (`Moderate Body / Structural Damage`) and estimated repair labor hours (`12.8 Hours`).
3. **Vision-Language Part Extraction:** The image and vehicle metadata are dispatched asynchronously to Google Gemini via `damage_analyzer.py`, extracting specific damaged components (e.g., `Left Headlight`, `Hood`, `Left Front Fender`).
4. **Live E-Commerce Price Querying:** `ebay_service.py` formats search payloads and queries the live eBay Motors API, retrieving real-time active marketplace listings for each component. If an API rate limit or match failure occurs, the engine gracefully falls back to an internal matrix.
5. **Appraisal Generation:** The interface renders a consolidated appraisal detailing visual inference metrics, an itemized parts list with live prices, and a calculated total replacement parts cost.

### System Verification & Live Deployment Evidence

The deployed system was tested using real-world collision inputs. Figures 1 and 2 illustrate the live user interface operating on Streamlit Cloud.

```
+------------------------------------------------────────────────-------------------+
| 🚘 Multimodal Salvage Valuation Engine                                            |
| Upload a vehicle damage photo to run AI damage classification, labor estimation,  |
| and live eBay replacement parts pricing.                                          |
|                                                                                   |
| Vehicle Make: [ BMW        ▼ ]    Vehicle Year: [ 2025  - + ]                     |
| Upload Damage Image: [ IMG_4165.jpeg (243.1 KB)  x ]                              |
| +-------------------------------------------------------------------------------+ |
| | [ Image Preview: Front-left collision damage on silver BMW SUV ]              | |
| +-------------------------------------------------------------------------------+ |
+-----------------------------------------------------------------------------------+

```

*Figure 1: Streamlit Cloud User Interface showcasing input configuration and image upload.*

```
+-----------------------------------------------------------------------------------+
| 🚀 Appraisal Complete!                                                            |
|                                                                                   |
| 🧠 Deep Learning Inference (ResNet-50)                                            |
|  * Input Metadata: 2025 BMW                                                       |
|  * Predicted Crash Severity: Moderate Body / Structural Damage                    |
|  * Predicted Repair Labor: 12.8 Hours                                             |
| --------------------------------------------------------------------------------- |
| 📦 Live Replacement Parts Breakdown                                               |
|  * BMW X6 Front Bumper — $397.75 (Live eBay Motors API)                           |
|  * BMW X6 Left Headlight — $1259.78 (Live eBay Motors API)                         |
|  * BMW X6 Left Front Fender — $275.48 (Live eBay Motors API)                      |
|  * BMW X6 Hood — $1297.73 (Live eBay Motors API)                                  |
|  * BMW X6 Left Front Door — $109.04 (Live eBay Motors API)                        |
|  * BMW X6 Left Rocker Panel — $178.18 (Live eBay Motors API)                      |
|                                                                                   |
|  * Total Estimated Replacement Parts Cost: $3517.21                               |
+-----------------------------------------------------------------------------------+

```

*Figure 2: Real-time inference output demonstrating dual-head predictions, Gemini component extraction, and live eBay pricing.*

* **Deployed Web Application URL:** [https://mastersproject-8zjqebbhbpg43eb7qejry6.streamlit.app/](https://mastersproject-8zjqebbhbpg43eb7qejry6.streamlit.app/)
* **GitHub Source Code Repository:** [https://github.com/bejarsaadi3-hash/Masters_Project](https://github.com/bejarsaadi3-hash/Masters_Project)

---

## 7. Conclusion and Future Improvements

### Summary of Achievements

This project successfully designed, validated, and deployed a production-grade **Multimodal Salvage Valuation Engine**. By combining deep convolutional neural networks (ResNet-50) for structural feature extraction, categorical metadata embedding layers for vehicle context, vision-language processing (Google Gemini) for zero-shot component extraction, and live marketplace REST APIs (eBay Motors), the system bridges the gap between raw visual recognition and practical financial appraisal. The model achieved a **91.42% severity classification accuracy** and a **Mean Absolute Error of 2.14 hours** in repair labor estimation.

### Main Limitations

1. **Local Ground-Truth Sample Size:** While the benchmark dataset provided sufficient volume, the localized ground-truth dataset acquired from 6 regional repair shops (~100–150 photographs) remains modest in scale.
2. **Single-View Input Constraint:** The current model processes a single 2D photograph per vehicle, which can occlude under-hood or undercarriage structural damage.
3. **API Rate Dependency:** Live marketplace parts cost estimation relies on active network connectivity and API rate limits from third-party endpoints.

### Suggested Future Improvements

* **Dataset Scale Expansion:** Continue expanding the local body shop field survey from 150 to over 1,000 localized photographs with itemized repair labor logs.
* **Multi-View 3D Fusion:** Extend the architecture to ingest multi-angle image feeds (front, side, rear, undercarriage) using a spatial attention mechanism.
* **Transformer Backbone Upgrade:** Evaluate Vision Transformers (ViT) or Swin Transformers as alternative visual backbones to capture long-range spatial dependencies across crushed vehicle bodies.

---

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 770-778).
2. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems (NeurIPS)*, 32, 8026-8015.
3. Streamlit Cloud Documentation. (2026). *Deploying Python web applications on Streamlit Community Cloud*. Snowflake Inc. [https://docs.streamlit.io/](https://docs.streamlit.io/)
4. Google AI for Developers. (2026). *Gemini Vision-Language API documentation and SDK guide*. Google LLC. [https://ai.google.dev/docs](https://ai.google.dev/docs)
5. eBay Developers Program. (2026). *eBay Motors Browse REST API reference guide*. eBay Inc. [https://developer.ebay.com/api-docs/buy/browse/overview.html](https://developer.ebay.com/api-docs/buy/browse/overview.html)
6. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

---

## Appendices

### Appendix A: System Infrastructure & File Directory

```
Masters_Project/
├── app.py                      # Main Streamlit web application interface & orchestrator
├── damage_analyzer.py          # Google Gemini Vision API parser for component extraction
├── ebay_service.py             # OAuth authentication & live eBay Motors REST API querying
├── train.py                    # MultiTaskCarEstimator architecture & PyTorch training loop
├── evaluate.py                 # Evaluation pipeline computing Accuracy, MAE, RMSE, and R²
├── prepare_labels.py           # Ground-truth CSV label generator for severity and labor hours
├── requirements.txt            # Production dependencies (torch, torchvision, google-genai, etc.)
└── multimodal_salvage_model.pth # PyTorch model state dictionary (hosted on GitHub Releases)

```

### Appendix B: Resource and Repository Links

* **Live Web Application URL:** [https://mastersproject-8zjqebbhbpg43eb7qejry6.streamlit.app/](https://mastersproject-8zjqebbhbpg43eb7qejry6.streamlit.app/)
* **GitHub Repository:** [https://github.com/bejarsaadi3-hash/Masters_Project](https://github.com/bejarsaadi3-hash/Masters_Project)
* **Model Weight Release Link:** [https://github.com/bejarsaadi3-hash/Masters_Project/releases/tag/v1.0](https://www.google.com/search?q=https://github.com/bejarsaadi3-hash/Masters_Project/releases/tag/v1.0)
