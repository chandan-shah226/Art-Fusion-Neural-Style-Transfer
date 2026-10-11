
# ART FUSION: REAL-TIME ARBITRARY NEURAL STYLE TRANSFER WITH ADAPTIVE INSTANCE NORMALIZATION AND DYNAMIC TEXT PRESERVATION
*(Internal Project Moniker: CanvasFlow)*

**A Final-Year Project Report Submitted in Partial Fulfillment of the Requirements for the Degree of Bachelor of Technology / Bachelor of Engineering in Computer Science and Engineering**

---

## PROJECT METADATA & REVISION RECORD

- **Project Title:** Art Fusion: Real-Time Arbitrary Neural Style Transfer with Adaptive Instance Normalization and Dynamic Text Preservation
- **Repository / Directory Code:** `ai-nst-project` / `NST_Code`
- **Candidate Name / Roll No:** [Candidate Name] / [Registration / Roll Number]
- **Academic Supervisor / Guide:** [Supervisor Name, Designation, Department]
- **Institutional Department:** Department of Computer Science and Engineering
- **Academic Year:** 2025–2026
- **Document Version:** 2.0 (Consolidated & Revised Final Submission)
- **Document Standard:** A4 Format, Cambria 12 pt equivalent typography, 1.5 line spacing, chapter-wise numbering.

---

## TABLE OF CONTENTS

- **List of Figures**
- **List of Tables**
- **List of Abbreviations**
- **Chapter 1: Introduction**
  - 1.1 Background of the Project
  - 1.2 Problem Statement
  - 1.3 Need and Motivation
  - 1.4 Objectives
  - 1.5 Scope and Constraints (Scope Matrix)
  - 1.6 Expected vs. Achieved Outcomes
  - 1.7 Organization of the Report
- **Chapter 2: Organization / Internship / Training Profile**
  - 2.1 Host Department and Work Environment
  - 2.2 Roles and Assigned Responsibilities
  - 2.3 Development Tools and Technology Stack
  - 2.4 Project Execution Schedule
- **Chapter 3: Literature Review**
  - 3.1 Evolution of Neural Style Transfer
  - 3.2 Critical Review of Related Work
  - 3.3 Comparative Technical Matrix
  - 3.4 Research Gap and Justification
- **Chapter 4: System Requirements and Feasibility Analysis**
  - 4.1 System Overview
  - 4.2 Software Requirements
  - 4.3 Hardware Requirements (Component, Specification, Purpose, Availability)
  - 4.4 Feasibility Study (Technical, Operational, Economic)
- **Chapter 5: System Architecture and Methodology**
  - 5.1 End-to-End Architectural Pipeline
  - 5.2 Pre-trained VGG-19 Feature Encoder
  - 5.3 Adaptive Instance Normalization (AdaIN) Formulation
  - 5.4 Feature-Space Linear Interpolation ($\alpha$ Slider)
  - 5.5 Symmetrical Convolutional Decoder Design
  - 5.6 Dynamic Text Preservation Pipeline (EasyOCR & Feathered Masking)
- **Chapter 6: Implementation Details**
  - 6.1 Project Directory and Module Architecture
  - 6.2 Training Workflow and Loss Formulations
  - 6.3 Hyperparameter Setup and Dataset Ingestion
  - 6.4 Flask Web Application and UI Implementation
- **Chapter 7: Results and Discussion**
  - 7.1 Visual Quality and Arbitrary Style Adaptation
  - 7.2 Style Intensity Control ($\alpha$ Interpolation Results)
  - 7.3 Text Legibility Preservation Evaluation
  - 7.4 Latency and Inference Benchmarks (CPU vs. GPU)
  - 7.5 Verification and Functional Test Cases
- **Chapter 8: Limitations and Future Scope**
  - 8.1 Current Technical Limitations
  - 8.2 Recommended Future Work
- **Chapter 9: Conclusion**
  - 9.1 Summary of Contributions
  - 9.2 Key Viva Defense Takeaways
- **References**
- **Appendices**
  - Appendix A: Core AdaIN and Statistical Alignment Code
  - Appendix B: Dynamic Text Blending Logic

---

## LIST OF FIGURES

| Figure Number | Caption / Description | Reference Section |
| :--- | :--- | :--- |
| **Figure 5.1** | End-to-End System Pipeline of the Art Fusion System | Section 5.1 |
| **Figure 5.2** | VGG-19 Encoder Layer Decomposition (`enc_1` to `enc_4`) | Section 5.2 |
| **Figure 5.3** | Adaptive Instance Normalization Statistical Alignment Scheme | Section 5.3 |
| **Figure 5.4** | Symmetrical 4-Stage Convolutional Decoder Architecture | Section 5.5 |
| **Figure 5.5** | Dynamic Text Preservation and Gaussian Feathered Blending Pipeline | Section 5.6 |
| **Figure 7.1** | Style Transfer Demonstrations Across Classical Artwork Styles | Section 7.1 |
| **Figure 7.2** | Linear Interpolation Progression with Style Slider ($\alpha \in [0.0, 1.0]$) | Section 7.2 |
| **Figure 7.3** | Comparative Typography Preservation: Standard AdaIN vs. Art Fusion | Section 7.3 |

---

## LIST OF TABLES

| Table Number | Caption / Description | Reference Section |
| :--- | :--- | :--- |
| **Table 1.1** | Project Scope and Boundary Matrix | Section 1.5 |
| **Table 1.2** | Expected Plan vs. Final Achieved Deliverables | Section 1.6 |
| **Table 2.1** | Project Execution Milestones and Deliverables | Section 2.4 |
| **Table 3.1** | Comparative Evaluation of Neural Style Transfer Techniques | Section 3.3 |
| **Table 4.1** | Software Environment and Framework Specifications | Section 4.2 |
| **Table 4.2** | Hardware Requirements, Purpose, and Availability Matrix | Section 4.3 |
| **Table 6.1** | Model Training Hyperparameters and Configurations | Section 6.3 |
| **Table 7.1** | Quantitative Inference Latency Across Processing Units | Section 7.4 |
| **Table 7.2** | Functional Integration Test Suite Summary | Section 7.5 |

---

## LIST OF ABBREVIATIONS

- **AdaIN:** Adaptive Instance Normalization
- **BN:** Batch Normalization
- **CNN:** Convolutional Neural Network
- **CRAFT:** Character Region Awareness for Text Detection
- **CUDA:** Compute Unified Device Architecture
- **IN:** Instance Normalization
- **MSE:** Mean Squared Error
- **NST:** Neural Style Transfer
- **OCR:** Optical Character Recognition
- **PIL:** Python Imaging Library (Pillow)
- **ReLU:** Rectified Linear Unit
- **RGB:** Red, Green, Blue
- **VGG:** Visual Geometry Group (University of Oxford)
- **WCT:** Whitening and Coloring Transform
- **WTF:** Web Forms Framework for Flask

---
---

# CHAPTER 1
# INTRODUCTION

## 1.1 Background of the Project
Neural Style Transfer (NST) is an established computer vision discipline that merges the visual structure of a content image with the brushstrokes, color distributions, and textures of an artistic style reference. The field gained prominence through the work of Gatys et al. (2015), who showed that deep convolutional representations learned by a pre-trained VGG network can decouple spatial structure from visual appearance. 

However, early optimization-based NST required hundreds of gradient descent steps per image, taking several minutes even on high-end GPUs. Subsequent feed-forward models achieved sub-second inference speeds, but they suffered from severe rigidity: each trained network was tied to exactly one artistic style, requiring separate training for every new painting.

In 2017, Huang and Belongie introduced **Adaptive Instance Normalization (AdaIN)**, establishing that artistic style corresponds directly to the channel-wise mean and variance of intermediate convolutional feature maps. By shifting and scaling normalized content activations with style statistics in a single feed-forward pass, a single decoder network can stylize images using unseen artwork without retraining.

This project, titled **"Art Fusion: Real-Time Arbitrary Neural Style Transfer with Adaptive Instance Normalization and Dynamic Text Preservation"** (developed under the internal project name **CanvasFlow**), implements this architecture in PyTorch and integrates it into a production-ready Flask web platform. Additionally, the project resolves a major flaw in practical NST workflows: painterly distortion of text. By integrating EasyOCR with a dynamic alpha-weighted feathered blending algorithm, the system automatically protects typographic readability in posters, book covers, and signage.

---

## 1.2 Problem Statement
Traditional neural style transfer systems present three critical engineering challenges:
1. **Iterative Latency:** Optimization-based algorithms require minutes per image, preventing interactive web deployment.
2. **Limited Style Generalization:** First-generation feed-forward networks require retraining and separate model files for every new artistic style.
3. **Typographic Breakdown:** Standard convolutional networks treat text, signs, and logos as raw texture patterns. The resulting painterly strokes distort letterforms, rendering text unreadable unless users manually paint masks.

**Formalized Problem Statement:**  
> *To design, implement, and deploy an interactive, real-time style transfer system capable of transferring arbitrary artistic styles using a single deep learning model, while providing dynamic intensity control and automatically preserving the legibility of textual elements in the content image.*

---

## 1.3 Need and Motivation
- **Accessible Creative Automation:** Designers, educators, and content creators need quick aesthetic styling without spending hours drawing manual layer masks in graphic design software.
- **Resource-Conscious Deep Learning:** A feed-forward AdaIN network executes fast inference on consumer-grade hardware and local workstations without demanding multi-GPU cloud clusters.
- **Automated Text Protection:** Graphic assets such as posters, flyers, album art, and road photography require legible lettering alongside artistic styling. Automating text extraction removes the burden of manual masking.
- **Practical Engineering Rigor:** Building the end-to-end pipeline from mathematical feature normalization in PyTorch to web deployment bridges theoretical deep learning and practical application design.

---

## 1.4 Objectives
1. **Construct the AdaIN Architecture:** Build a pre-trained VGG-19 feature encoder truncated at `relu4_1` with frozen weights, an AdaIN statistical transfer layer, and a 4-stage convolutional decoder using reflection padding.
2. **Train the Convolutional Decoder:** Train the decoder on MS-COCO 2017 and WikiArt datasets using a joint perceptual loss balancing content reconstruction at `relu4_1` and multi-scale style matching across layers `relu1_1`, `relu2_1`, `relu3_1`, and `relu4_1`.
3. **Enable Dynamic Style Intensity ($\alpha$ Slider):** Implement linear feature interpolation between unstylized content features and stylized features, allowing continuous user adjustment from $0.0$ to $1.0$.
4. **Automate Typographic Preservation:** Integrate EasyOCR to detect horizontal and polygonal text regions and apply a Gaussian feathered mask modulated by the user's $\alpha$ setting.
5. **Develop and Deploy an Interactive Web Interface:** Build a responsive Flask application supporting drag-and-drop uploads, 16 curated artistic presets, real-time controls, and CPU/GPU auto-detection.

---

## 1.5 Scope and Constraints

### Table 1.1: Project Scope and Boundary Matrix
| Dimension | In Scope | Out of Scope | Constraints |
| :--- | :--- | :--- | :--- |
| **Input Media** | Static RGB images (PNG, JPG, JPEG). | Video streams, animated GIFs, raw vectors. | Resized to 512px during inference to balance memory and latency. |
| **Model Scope** | Arbitrary style transfer via frozen VGG-19 encoder and trained AdaIN decoder. | Photorealistic style transfer; training custom CNN backbones from scratch. | VGG weights are permanently frozen; only the decoder network is trained. |
| **Style Control** | Global linear feature interpolation ($\alpha \in [0.0, 1.0]$ in increments of 0.05). | Manual local brush painting of styles on user-selected regions. | Linear interpolation operates globally across feature maps. |
| **Text Handling** | Automated English text detection (EasyOCR) with alpha-responsive feathered blending. | Multilingual non-Latin script parsing; manual polygon mask editing tools. | Detection accuracy depends on font contrast and image resolution. |
| **System Design**| Flask web service, WTForms validation, responsive CSS, Jinja2 templating. | Multi-user login accounts, persistent relational databases, cloud container clusters. | Operates statelessly with temporary storage in `static/uploads/`. |
| **Environment** | PyTorch 2.x on Windows / Linux with automatic CUDA/CPU device selection. | Embedded mobile microcontrollers or mobile DSP chips. | Sub-second inference on GPU; 2–4 seconds on multi-core CPU. |

---

## 1.6 Expected vs. Achieved Outcomes

### Table 1.2: Expected Plan vs. Final Achieved Deliverables
| Metric / Feature | Expected Outcome (Initial Plan) | Final Achieved Outcome | Remarks |
| :--- | :--- | :--- | :--- |
| **Inference Latency** | Under 5 seconds per image. | < 1 second on GPU; ~2.5–3.5s on CPU. | Real-time execution via feed-forward AdaIN. |
| **Style Versatility** | Arbitrary style transfer without retraining. | Fully arbitrary transfer across unseen artworks. | Validated over 16 curated artistic presets. |
| **Model Architecture** | VGG encoder + AdaIN layer + Decoder. | Frozen VGG-19 (`relu4_1`) + AdaIN + 4-stage convolutional decoder. | Implemented and verified in `models.py`. |
| **Loss Formulation** | Balanced content and style losses. | MSE content loss at `relu4_1` and multi-layer MSE style loss. | Balanced using style weight $\lambda_s = 10$. |
| **Transfer Strength** | Adjustable style intensity. | Responsive slider adjusting $\alpha \in [0.0, 1.0]$. | Seamless linear interpolation in feature space. |
| **Text Preservation** | Not in original baseline plan. | Automated EasyOCR detection with dynamic alpha-weighted feathering. | Added as a major usability enhancement. |
| **User Interface** | Basic upload form and result view. | Modern web UI with presets, drag-and-drop upload, and instant download. | Responsive layout with client validation. |
| **Storage Architecture**| Optional database. | Stateless storage in `static/uploads/`. | Clean design without unnecessary database overhead. |

---

## 1.7 Organization of the Report
- **Chapter 2:** Summarizes the training profile, laboratory environment, roles, and project milestones.
- **Chapter 3:** Reviews foundational NST literature, comparative techniques, and the identified research gap.
- **Chapter 4:** Details software, hardware requirements, and technical/economic feasibility.
- **Chapter 5:** Explains the end-to-end architecture, mathematical AdaIN formulations, decoder design, and OCR pipeline.
- **Chapter 6:** Outlines implementation specifics, loss functions, training parameters, and Flask routing.
- **Chapter 7:** Analyzes experimental results, style variations, text legibility, latency, and test validation.
- **Chapter 8:** Discusses current technical limitations and concrete directions for future work.
- **Chapter 9:** Summarizes key contributions and core viva defense concepts.

---
---

# CHAPTER 2
# ORGANIZATION / INTERNSHIP / TRAINING PROFILE

## 2.1 Host Department and Work Environment
This project was carried out as part of the **Final-Year Engineering Project Curriculum** in the **Department of Computer Science and Engineering** [or affiliated training laboratory: *Centre for Artificial Intelligence and Deep Learning Solutions*].

The host laboratory focuses on applied machine learning, computer vision, and deployable software engineering. The working environment followed an agile development structure with weekly milestone reviews, training loss evaluations, code inspections, and structured testing on local workstations.

---

## 2.2 Roles and Assigned Responsibilities
As the primary project developer, core responsibilities included:
1. **Literature Survey:** Evaluating optimization-based, feed-forward, and feature normalization approaches to establish the technical foundation.
2. **Deep Learning Engineering:** Implementing the VGG-19 feature encoder, the AdaIN mathematical alignment function, and the symmetrical convolutional decoder in PyTorch.
3. **Training Pipeline Configuration:** Preparing MS-COCO 2017 and WikiArt data pipelines, writing composite loss computations, tuning training hyperparameters, and logging loss metrics.
4. **Text Preservation Module:** Designing and integrating the automated EasyOCR pipeline with Gaussian feathered blending modulated by the style strength parameter $\alpha$.
5. **Full-Stack Web Development:** Developing a responsive Flask application with WTForms validation, image uploads, preset style management, and cross-platform CPU/GPU execution.
6. **Testing and Academic Documentation:** Executing automated functional test suites (`test_app.py`, `test_full.py`) and compiling formal documentation.

---

## 2.3 Development Tools and Technology Stack
- **Operating Environment:** Windows 11 / Ubuntu 22.04 LTS (64-bit)
- **Deep Learning Framework:** PyTorch 2.x, Torchvision
- **OCR Engine:** EasyOCR (PyTorch-backed text detection and recognition)
- **Image Processing Libraries:** Pillow (PIL), OpenCV, NumPy
- **Web Application Backend:** Flask 3.1, Flask-WTF, WTForms, Werkzeug, Jinja2
- **Frontend Technologies:** HTML5, CSS3, JavaScript, FontAwesome, Google Fonts
- **Development Tooling:** Visual Studio Code, Git, GitHub, Jupyter Notebook

---

## 2.4 Project Execution Schedule

### Table 2.1: Project Execution Milestones and Deliverables
| Phase | Duration | Core Tasks and Milestones | Deliverables |
| :--- | :--- | :--- | :--- |
| **Phase 1: Planning & Literature Survey** | Weeks 1–3 | Surveyed NST algorithms; identified typography breakdown in artistic transfer; finalized project scope. | Literature synthesis report; project specification. |
| **Phase 2: Architecture Implementation** | Weeks 4–7 | Constructed VGG-19 encoder, AdaIN layer, and symmetrical decoder in PyTorch (`models.py`, `utils.py`). | Verified PyTorch model classes and tensor forward passes. |
| **Phase 3: Dataset Setup & Training** | Weeks 8–11 | Ingested MS-COCO and WikiArt datasets; configured loss functions; trained decoder for 200 epochs (`train.py`). | Trained weights checkpoint (`decoder_final.pth`). |
| **Phase 4: OCR Pipeline Integration** | Weeks 12–13 | Integrated EasyOCR; engineered Gaussian feathered blending and alpha-responsive scaling (`app.py`). | Functional text-preservation masking module. |
| **Phase 5: Web App & Testing** | Weeks 14–15 | Developed Flask interface, drag-and-drop UI, preset gallery; ran unit and integration tests. | Responsive web application; test scripts (`test_app.py`). |
| **Phase 6: Final Review & Viva Prep** | Week 16 | Conducted CPU/GPU latency benchmarks; finalized academic documentation and viva defense materials. | Consolidated final project report. |

---
---

# CHAPTER 3
# LITERATURE REVIEW

## 3.1 Evolution of Neural Style Transfer
Neural Style Transfer originated from the realization that deep Convolutional Neural Networks pre-trained on object classification (such as VGG-19 on ImageNet) act as hierarchical feature extractors. Early convolutional layers (`conv1`, `conv2`) capture localized edges, textures, and color patches. Deeper convolutional layers (`conv4`, `conv5`) capture complex semantic structures, object geometry, and spatial arrangements while discarding exact pixel values.

Stylization involves retaining high-level semantic activations of a content image while altering low-level texture statistics to reflect a style reference image. The progression of NST can be divided into three generations:
1. **Optimization-based methods (2015):** Iteratively optimize image pixels through backpropagation. Flexible across styles, but computationally slow.
2. **Single-style feed-forward networks (2016):** Train a generative network per style. Fast inference, but cannot generalize to unseen styles without complete retraining.
3. **Universal / Arbitrary feature-transform networks (2017–Present):** Dynamically manipulate deep feature statistics at runtime via statistical normalization (AdaIN) or matrix decomposition (WCT). Fast and capable of applying arbitrary unseen styles.

---

## 3.2 Critical Review of Related Work

### 1. Gatys, Ecker, and Bethge (CVPR 2016) — *Image Style Transfer Using Convolutional Neural Networks*
Gatys et al. formulated NST as an optimization task. Content was defined using feature activations from layer `relu4_2`, while style was defined using Gram matrices (channel-wise inner products) computed across layers `relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`, and `relu5_1`. Using L-BFGS gradient descent, the algorithm updated pixel values directly. While delivering compelling painterly quality, the method required hundreds of iterations (taking 1 to 5 minutes per image), making real-time applications impractical.

### 2. Johnson, Alahi, and Fei-Fei (ECCV 2016) — *Perceptual Losses for Real-Time Style Transfer*
Johnson et al. replaced pixel optimization with an image transformation feed-forward network trained using perceptual loss functions from a frozen VGG network. Once trained, the network performed stylization in a single forward pass (< 0.5s). However, each network was strictly tied to a single artistic style. Supporting multiple styles required training and storing separate multi-megabyte model checkpoints for each artwork.

### 3. Ulyanov, Vedaldi, and Lempitsky (2016) — *Instance Normalization: The Missing Ingredient for Fast Stylization*
Ulyanov et al. observed that Batch Normalization retains batch-dependent contrast information that interferes with stylization. Replacing Batch Normalization with Instance Normalization (normalizing each image channel independently across spatial dimensions) significantly improved visual stylization quality and accelerated convergence. This demonstrated that image style is strongly tied to the spatial mean and standard deviation of feature maps.

### 4. Huang and Belongie (ICCV 2017) — *Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization*
Building upon Instance Normalization, Huang and Belongie introduced Adaptive Instance Normalization (AdaIN). AdaIN shifts and scales the normalized feature activations of a content image to match the mean and standard deviation of an arbitrary style image. A single convolutional decoder, trained on a large content and style corpus, inverts the aligned features back to an RGB image in a single pass. This solved the dual challenges of latency and style generalization.

### 5. Li et al. (NeurIPS 2017) — *Universal Style Transfer via Feature Transforms (WCT)*
Li et al. proposed Whitening and Coloring Transforms (WCT) to align feature covariance matrices using Singular Value Decomposition (SVD). While WCT captures multi-level correlations without requiring custom loss weighting during decoder training, matrix SVD operations at runtime are computationally expensive, resulting in higher latency compared to the lightweight mean-and-variance alignment of AdaIN.

### 6. Baek et al. (CVPR 2019) — *Character Region Awareness for Text Detection (CRAFT / EasyOCR)*
Baek et al. developed CRAFT, which detects scene text by predicting character regions and affinities between characters. Implemented within EasyOCR, this network detects horizontal, rotated, and polygonal text across complex backgrounds. In this project, EasyOCR provides the foundation for identifying text regions on content images to prevent painterly corruption.

---

## 3.3 Comparative Technical Matrix

### Table 3.1: Comparative Evaluation of Neural Style Transfer Techniques
| Technical Dimension | Optimization Model (Gatys et al.) | Single-Style Feed-Forward (Johnson et al.) | Proposed Art Fusion System (AdaIN + Text Masking) |
| :--- | :--- | :--- | :--- |
| **Inference Latency** | 1 to 5 minutes per image (slow). | 0.1 to 0.5 seconds on GPU. | **0.3 to 0.8 seconds on GPU; ~3s on CPU.** |
| **Style Generalization**| Arbitrary styles supported. | Strictly 1 style per trained network. | **Arbitrary styles via a single universal model.** |
| **Model Disk Footprint**| No decoder; requires full VGG network. | 7 MB to 20 MB per style. | **Single ~14 MB universal decoder checkpoint.** |
| **Intensity Control ($\alpha$)**| Requires re-running optimization. | Fixed during network training. | **Continuous real-time slider ($\alpha \in [0.0, 1.0]$).** |
| **Typographic Handling**| Blurs and distorts text into paint strokes. | Blurs and distorts text into paint strokes. | **Automated EasyOCR detection with dynamic blending.** |
| **Deployment Suitability**| Impractical for interactive web apps. | Moderate (limited by preset count). | **High: lightweight Flask app with CPU/GPU support.** |

---

## 3.4 Research Gap and Justification
The literature review highlights two primary gaps in practical style transfer systems:
1. **The Typography Degradation Gap:** Mainstream NST systems assume content images consist purely of natural scenes or portraits. Real-world images—such as advertisements, event banners, posters, and book covers—frequently include text. Convolutions treat letters as high-contrast edges and blend them into the artistic style, rendering them illegible. Existing solutions required users to manually paint masks in photo editors.
2. **Accessible Web Integration:** Existing open-source AdaIN implementations are largely research-oriented command-line scripts requiring manual GPU configuration. There is a lack of compact, self-contained web applications that offer arbitrary style selection, real-time intensity control, and automated text protection.

**Project Justification:**  
This project addresses these gaps by combining the real-time efficiency of **Adaptive Instance Normalization**, an automated **EasyOCR-based feathered blending pipeline**, and a **lightweight Flask web application** that runs on both consumer GPUs and standard CPUs.

---
---

# CHAPTER 4
# SYSTEM REQUIREMENTS AND FEASIBILITY ANALYSIS

## 4.1 System Overview
The system is built as a modular client-server web application. The frontend allows users to upload content images, pick from preset artworks or upload custom style references, and adjust style intensity via an interactive slider. The backend server runs on Flask, executing deep learning inference via PyTorch and character extraction via EasyOCR.

## 4.2 Operating Environment & Architecture
The system operates as a hybrid deep learning and web application environment. The core training and inference workflows execute on top of Python and PyTorch, interfacing with EasyOCR for computer vision text localization and Flask for HTTP request dispatching.

---

## 4.3 Hardware Requirements (if applicable)

### Table 4.1: Hardware Requirements, Purpose, and Availability Matrix
| Component | Specification | Purpose | Availability |
| :--- | :--- | :--- | :--- |
| **Processor (CPU)** | Intel Core i5 / Core i7 or AMD Ryzen 5 / 7 (Quad-Core+, 2.5 GHz+) | Manages Flask server routing, image preprocessing / resizing, and fallback inference. | **Available** (Standard local workstation hardware) |
| **Dedicated Graphics (GPU)** | NVIDIA GeForce GTX 1650 / RTX 2060 / RTX 3060+ (4 GB+ VRAM, CUDA) | Accelerates deep learning tensor operations, multi-epoch training, and sub-second inference. | **Available** (Workstation GPU for training; optional on CPU deployment) |
| **System Memory (RAM)** | 8 GB minimum (16 GB DDR4/DDR5 recommended) | Accommodates PyTorch model weights (`vgg_normalised.pth`, `decoder_final.pth`) and image buffers. | **Available** (Workstation standard configuration) |
| **Secondary Storage (SSD)** | 10 GB to 20 GB free space (NVMe / SATA SSD preferred) | Stores training datasets (MS-COCO, WikiArt), saved checkpoints, uploaded files, and logs. | **Available** (Local SSD storage) |
| **Display Interface** | 1080p Full HD Display (1920 × 1080 resolution, 24-bit True Color) | Facilitates UI design, side-by-side visual evaluation, artifact inspection, and viva presentations. | **Available** (Standard desktop display) |
| **Input Devices** | Standard QWERTY Keyboard and 3-button Optical Mouse | Code authoring, CLI execution, image upload selection, and interactive slider tuning. | **Available** (Standard peripheral hardware) |
| **Network Interface** | 10/100/1000 Mbps Ethernet / Wi-Fi | Ingesting pre-trained models, Python package dependencies, and external reference images. | **Available** (Campus / home broadband connection) |

---

## 4.4 Software Requirements

### Table 4.2: Software Requirements, Version, Purpose, and License Matrix
| Software / Tool | Version | Purpose | Source / License |
| :--- | :--- | :--- | :--- |
| **Operating System** | Windows 10/11 / Ubuntu 20.04+ (64-bit) | Host runtime operating system for training, development, and web hosting. | Microsoft / Canonical (Open Source) |
| **Python** | 3.9+ / 3.12 | Core programming runtime for ML models, data pipelines, and web backend. | Python Software Foundation (PSF License) |
| **PyTorch (`torch`)** | 2.0.0+ | Deep learning framework: tensor computations, autograd, and GPU execution. | Linux Foundation / Modified BSD License |
| **Torchvision** | 0.15.0+ | Computer vision utilities, image tensor transformations (`Resize`, `ToTensor`). | BSD 3-Clause License |
| **EasyOCR** | 1.7.0+ | Deep learning scene text detection (CRAFT) and character region localization. | Jaided AI / Apache License 2.0 |
| **Flask** | 3.0+ / 3.1 | Lightweight WSGI web framework for HTTP request routing and template rendering. | Pallets Projects / BSD 3-Clause License |
| **Flask-WTF / WTForms**| Flask-WTF 1.2+, WTForms 3.0+ | Secure web form handling, CSRF protection, and file upload validation. | BSD 3-Clause License |
| **Pillow (PIL)** | 9.0+ / 10.0+ | Image loading, format conversion, Gaussian blur filtering, and alpha compositing. | Historical Permission Notice & Disclaimer (HPND) |
| **NumPy** | 1.22+ / 2.0+ | Multidimensional array manipulation and vectorized numerical operations. | NumPy Developers / BSD 3-Clause License |
| **Werkzeug** | 3.0+ | Secure filename sanitation (`secure_filename`) and WSGI request utilities. | Pallets Projects / BSD 3-Clause License |
| **NVIDIA CUDA Toolkit** | 11.8 / 12.x | Parallel computing architecture and cuDNN library for GPU deep learning. | NVIDIA Corporation (Free for Developers) |
| **Visual Studio Code** | 1.85+ | Primary Integrated Development Environment (IDE) for coding and debugging. | Microsoft (MIT License) |
| **Git & GitHub** | 2.40+ | Distributed version control system and repository tracking. | GNU GPL v2 / GitHub Web Platform |

---

## 4.5 Data / Dataset Requirements (if applicable)

### Table 4.3: Data / Dataset Specifications, Pre-processing, and Purpose
| Data Source | Type | Size / Records | Pre-processing | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **MS-COCO 2017 (`coco2017test`)** | Natural photographic RGB images | ~40,000 images (~6.5 GB) | Scaled to 512px, randomly cropped to 256×256, normalized, converted to PyTorch float tensors. | Serves as the content training corpus; teaches the decoder to reconstruct natural object geometry and spatial structures. |
| **WikiArt Dataset (`style_dataset`)** | Fine art digital paintings (various periods) | ~35,000 images (~7.2 GB) | Scaled to 512px, randomly cropped to 256×256, normalized, converted to PyTorch float tensors. | Serves as the style training corpus; exposes the AdaIN layer to diverse artistic textures, color palettes, and painterly strokes. |
| **Preset Art Gallery & Uploads (`examples/`, `uploads/`)** | Digital RGB image files (JPG, PNG) | 16 curated artwork presets + live user uploads (~15 MB) | Validated via WTForms, resized to 512px max dimension, EasyOCR text bounding box extraction, Gaussian feathering. | Provides interactive style presets in the web interface and enables real-time stylization testing during evaluation. |

---

## 4.6 Feasibility Study
- **Technical Feasibility:** The architecture relies on mature, battle-tested open-source libraries: PyTorch for tensor computation and EasyOCR for text detection. The feed-forward AdaIN network eliminates iterative optimization, ensuring the technical pipeline is reliable and fully deployable on consumer-grade hardware.
- **Operational Feasibility:** The web interface is designed for simplicity. Users do not need technical knowledge of neural networks or manual image editing; they simply upload an image, adjust a slider, and click a button. Automated text protection runs transparently in the background.
- **Economic & Environmental Feasibility:** The project uses freely available open-source software and open datasets (MS-COCO, WikiArt). By avoiding iterative GPU optimization and large multi-billion parameter architectures, training requires modest compute, and inference runs locally without ongoing cloud API expenses.

---
---

# CHAPTER 5
# SYSTEM ARCHITECTURE AND METHODOLOGY

## 5.1 End-to-End Architectural Pipeline
The Art Fusion system processes images through a synchronized deep learning and computer vision pipeline. The process involves three key stages: feature extraction, statistical style transfer, and dynamic text-preserved image reconstruction.

```
+-------------------------------------------------------------------------------------------------+
|                                    ART FUSION SYSTEM PIPELINE                                   |
+-------------------------------------------------------------------------------------------------+

 [Content Image (c)] ------> [Pre-trained VGG-19 Encoder (Frozen)] ------> Content Features f_c
                                                                                |
 [Style Image (s)]   ------> [Pre-trained VGG-19 Encoder (Frozen)] ------> Style Features f_s
                                                                                |
                                                                                v
                                                                   [AdaIN Statistical Alignment]
                                                                                |
                                                                                v
                                                                       Aligned Features t
                                                                                |
                                   User Alpha Slider (α) ----------> [Linear Feature Interpolation]
                                                                                |
                                                                                v
                                                                     Interpolated Features T
                                                                                |
                                                                                v
                                                                   [Convolutional Decoder]
                                                                                |
                                                                                v
                                                                      Stylized Image (I_stylized)
                                                                                |
 [Content Image (c)] ------> [EasyOCR Text Detection]                           |
                                      |                                         |
                                      v                                         |
                             [Feathered Gaussian Mask]                          |
                                      |                                         |
                                      +-------------------> [Dynamic Alpha Composite]
                                                                    |
                                                                    v
                                                         Final Output Image (I_out)
```
*Figure 5.1: End-to-End System Pipeline and Data Workflow (High-Resolution Diagram: [data_pipeline_workflow.png](file:///c:/Users/cvs02/OneDrive/Desktop/Projects/CLONE%20ADAIN/data_pipeline_workflow.png))*

*(Full high-resolution UML Activity & Sequence Workflow Diagram available at: [uml_workflow_user_system.png](file:///c:/Users/cvs02/OneDrive/Desktop/Projects/CLONE%20ADAIN/uml_workflow_user_system.png))*

---

## 5.2 Project Phases and Deliverables

### Table 5.1: Project Phases, Activities, Deliverables, and Evidence Matrix
| Phase | Activities | Deliverables | Evidence / Status |
| :--- | :--- | :--- | :--- |
| **Phase 1: Research, Architecture Selection & Mathematical Formulation** | Reviewed optimization vs. feed-forward style transfer; identified typography destruction problem; mathematically formulated AdaIN statistical normalization and VGG-19 encoder truncation at `relu4_1`. | Architecture specifications, mathematical formulation document, and repository layout. | **Completed**<br>(Documented in Literature Review, Chapter 3 comparative matrix, and `README.md`). |
| **Phase 2: PyTorch Model Implementation & Training Pipeline** | Constructed frozen VGG-19 encoder and 4-stage symmetrical convolutional decoder with reflection padding; configured MS-COCO 2017 & WikiArt data loaders; executed 200-epoch training minimizing composite perceptual loss ($\mathcal{L}_c + \lambda_s \mathcal{L}_s$). | PyTorch model definitions (`models.py`, `utils.py`), automated training pipeline (`train.py`), and trained weights checkpoint. | **Completed**<br>(Saved checkpoint `decoder_final.pth` [14.0 MB], training options log `options.txt`, and loss curves). |
| **Phase 3: Dynamic Text Preservation & Alpha Control Pipeline** | Integrated EasyOCR (CRAFT) to detect scene text bounding boxes and polygons; engineered Gaussian feathered masking ($r=3$); formulated dynamic $\alpha$-responsive blending weight ($w_{\text{text}}$); implemented bottleneck latent feature interpolation. | Automated text-preservation masking module and dynamic alpha slider integration in `app.py`. | **Completed**<br>(Verified preservation on text-heavy posters; operational logic in `create_text_mask()` within `app.py`). |
| **Phase 4: Full-Stack Web Deployment, Testing & Documentation** | Built responsive Flask application with CSRF protection, WTForms validation, and 16 preset artwork styles; implemented automatic CUDA/CPU device routing; wrote unit/integration tests; benchmarked latency; compiled final report. | Functional Flask web interface (`app.py`, `templates/`, `static/`), automated test suite (`test_app.py`, `test_full.py`), benchmarks, and report. | **Completed**<br>(Application operational at `localhost:5000`, test suite passing with HTTP 200, latency benchmarks documented). |

---

## 5.3 Project Schedule / Timeline

### Table 5.2: Project Schedule, Planned vs. Actual Timelines, and Status
| Task / Phase | Planned Start | Planned End | Actual Start | Actual End | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Problem Definition, Literature Survey & Architecture Design** | June 01, 2026 | June 21, 2026 | June 01, 2026 | June 20, 2026 | **Completed** |
| **Phase 2: PyTorch Model Engineering & Decoder Training (MS-COCO & WikiArt)** | June 22, 2026 | August 09, 2026 | June 22, 2026 | August 16, 2026 | **Completed** |
| **Phase 3: Dynamic Text Preservation Pipeline & Alpha Slider Integration** | August 10, 2026 | August 30, 2026 | August 17, 2026 | August 31, 2026 | **Completed** |
| **Phase 4: Full-Stack Web Deployment, Latency Profiling & Report Compilation** | September 01, 2026 | September 20, 2026 | September 01, 2026 | September 18, 2026 | **Completed** |

---

## 5.4 Pre-trained VGG-19 Feature Encoder
The feature extractor is based on the 19-layer VGG network pre-trained on ImageNet. Consistent with Huang and Belongie (2017), the network weights are normalized to ensure stable activation scales, and all parameters are permanently frozen (`requires_grad = False`).

The encoder is truncated at layer `relu4_1` (the 31st layer of VGG-19). In our implementation (`models.py`), the encoder is split into four sequential stages:
- `enc_1`: Layers up to `relu1_1` (64 feature channels)
- `enc_2`: Layers up to `relu2_1` (128 feature channels)
- `enc_3`: Layers up to `relu3_1` (256 feature channels)
- `enc_4`: Layers up to `relu4_1` (512 feature channels)

During inference, only `relu4_1` features are extracted ($f_c, f_s \in \mathbb{R}^{512 \times H' \times W'}$). During training, intermediate activations from all four stages are used to compute multi-layer style loss.

---

## 5.3 Adaptive Instance Normalization (AdaIN) Formulation
Adaptive Instance Normalization receives a content feature map $x$ and a style feature map $y$. Unlike Batch Normalization, which normalizes across batch instances, AdaIN normalizes each channel of each feature map independently across its spatial dimensions ($H \times W$), then applies the mean and standard deviation of the style feature map:

$$\text{AdaIN}(x, y) = \sigma(y) \left( \frac{x - \mu(x)}{\sigma(x)} \right) + \mu(y)$$

Where $\mu(x)$ and $\sigma(x)$ are the channel-wise mean and standard deviation:

$$\mu_c(x) = \frac{1}{HW} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{c,h,w}$$

$$\sigma_c(x) = \sqrt{\frac{1}{HW} \sum_{h=1}^{H} \sum_{w=1}^{W} \left( x_{c,h,w} - \mu_c(x) \right)^2 + \epsilon}$$

Here, $\epsilon = 10^{-5}$ prevents division by zero. AdaIN shifts and scales the normalized content activations, transferring the color and texture statistics of the style reference without modifying the spatial structure of the content image.

---

## 5.4 Feature-Space Linear Interpolation ($\alpha$ Slider)
To give users control over the transfer strength, the system performs linear interpolation directly in the bottleneck feature space before decoding:

$$T = \alpha \cdot \text{AdaIN}(f_c, f_s) + (1 - \alpha) \cdot f_c$$

- When $\alpha = 0.0$, $T = f_c$, and the decoder reconstructs the original unstylized content image.
- When $\alpha = 1.0$, $T = \text{AdaIN}(f_c, f_s)$, applying full artistic stylization.
- For intermediate values ($\alpha \in (0, 1)$), the output blends the structural fidelity of the content image with the texture of the style image.

Because this interpolation takes place in feature space rather than pixel space, it produces smooth, natural stylistic transitions without ghosting artifacts.

---

## 5.5 Symmetrical Convolutional Decoder Design
The decoder inverts the aligned feature representation $T \in \mathbb{R}^{512 \times H' \times W'}$ back into an RGB image $I \in \mathbb{R}^{3 \times H \times W}$. The decoder architecture mirrors VGG-19 up to `relu4_1`, replacing max-pooling layers with nearest-neighbor upsampling layers to eliminate grid artifacts:

1. **Stage 4:** $512 \to 256$ channels via $3 \times 3$ Conv + ReLU $\to$ Nearest Upsampling ($\times 2$).
2. **Stage 3:** Three $3 \times 3$ Conv layers (256 channels) + ReLU $\to$ Nearest Upsampling ($\times 2$).
3. **Stage 2:** Two $3 \times 3$ Conv layers (128 channels) + ReLU $\to$ Nearest Upsampling ($\times 2$).
4. **Stage 1:** One $3 \times 3$ Conv (64 channels) + ReLU $\to$ Final $3 \times 3$ Conv ($64 \to 3$ channels) producing the RGB image.

All convolutional layers use `ReflectionPad2d(1, 1, 1, 1)` rather than standard zero padding to avoid dark border artifacts along image edges.

---

## 5.6 Dynamic Text Preservation Pipeline
When style transfer is applied to posters, signs, or book covers, high-contrast letter boundaries are typically distorted into painterly brushstrokes. To address this, Art Fusion integrates an automated text preservation pipeline:

1. **Scene Text Detection:** EasyOCR processes the content image using its PyTorch-backed CRAFT detector, extracting horizontal bounding boxes $[x_{\min}, x_{\max}, y_{\min}, y_{\max}]$ and polygonal text boundaries.
2. **Mask Generation:** An 8-bit single-channel mask is initialized. Detected text bounding boxes are drawn with a padding of 4 pixels to enclose character serifs and ascenders.
3. **Gaussian Feathering:** The binary mask is softened using a Gaussian blur kernel ($\text{radius} = 3$). This creates a smooth gradient along text edges, avoiding harsh boundary lines during blending.
4. **Dynamic Alpha Modulation:** To ensure text appearance responds naturally to the user's style slider, the mask intensity is dynamically scaled:
   - **Legible Zone ($\alpha \le 0.5$):** Text is kept clear and readable:
     $$w_{\text{text}} = 1.0 - \left( \frac{\alpha}{0.5} \right) \times 0.20 \quad \implies w_{\text{text}} \in [1.0, 0.80]$$
   - **Artistic Blending Zone ($\alpha > 0.5$):** Text blends progressively into the artistic style:
     $$\text{progress} = \frac{\alpha - 0.5}{0.5}, \quad w_{\text{text}} = 0.80 - \text{progress} \times 0.70 \quad \implies w_{\text{text}} \in [0.80, 0.10]$$
5. **Alpha Compositing:** The final image is composited using Pillow:
   $$I_{\text{out}} = \text{Composite}(I_{\text{content}}, I_{\text{stylized}}, M_{\text{feathered}} \times w_{\text{text}})$$

If no text is detected, the stylized image is returned directly without additional processing overhead.

---
---

# CHAPTER 6
# IMPLEMENTATION DETAILS

## 6.1 Project Directory and Module Architecture
The project is organized in a modular structure within the workspace:

```
NST_Code/
├── app.py                      # Flask backend, routing, OCR masking, and inference
├── train.py                    # Multi-epoch PyTorch decoder training pipeline
├── vgg_normalised.pth          # Pre-trained VGG-19 encoder weights (frozen)
├── utils/
│   ├── models.py               # VGGEncoder (enc_1..enc_4) and Decoder neural network classes
│   └── utils.py                # AdaIN operator, calc_mean_std, ImageFolderDataset, transforms
├── templates/
│   └── index.html              # Jinja2 template with responsive UI, presets, and upload controls
├── static/
│   ├── css/style.css           # Custom stylesheets for the web dashboard
│   ├── js/app.js               # Client-side dynamic interaction script
│   └── uploads/                # Temporary local storage for uploaded and generated images
├── examples/                   # 16 curated artwork references and demonstration files
└── experiment/
    └── final_exp/              # Best checkpoint (decoder_final.pth) and options.txt
```

---

## 6.2 Training Workflow and Loss Formulations
During training, the VGG encoder weights remain frozen, and only the convolutional decoder is optimized. Training minimizes a composite perceptual loss function balancing content and style objectives:

$$\mathcal{L} = \mathcal{L}_c + \lambda_s \mathcal{L}_s$$

### 1. Content Loss ($\mathcal{L}_c$)
The content loss measures the Euclidean distance (MSE) between the target AdaIN feature map $t$ and the `relu4_1` features extracted from the decoder output $g = \text{Decoder}(t)$:

$$\mathcal{L}_c = \|\phi_{4,1}(g) - t\|_2^2$$

This encourages the decoder to invert the normalized feature representation back into image space accurately.

### 2. Style Loss ($\mathcal{L}_s$)
Rather than using computationally heavy Gram matrices, style loss is computed by comparing feature statistics (mean and standard deviation) across multiple layers of VGG-19:

$$\mathcal{L}_s = \sum_{i=1}^{4} \left( \|\mu(\phi_i(g)) - \mu(\phi_i(s))\|_2^2 + \|\sigma(\phi_i(g)) - \sigma(\phi_i(s))\|_2^2 \right)$$

Where $\phi_i$ denotes layers `relu1_1`, `relu2_1`, `relu3_1`, and `relu4_1`. This multi-scale formulation captures both fine textures (lower layers) and global color themes (higher layers).

---

## 6.3 Hyperparameter Setup and Dataset Ingestion

### Table 6.1: Model Training Hyperparameters and Configurations
| Parameter Name | Configuration Value | Technical Rationale / Source |
| :--- | :--- | :--- |
| **Content Dataset** | MS-COCO 2017 (`coco2017test`) | Diverse collection of natural images, objects, and scenes. |
| **Style Dataset** | WikiArt (`style_dataset`) | Large-scale collection of fine art covering diverse genres. |
| **Encoder Weights** | `vgg_normalised.pth` (80.1 MB) | Normalized VGG-19 pre-trained on ImageNet. |
| **Batch Size** | 16 | Selected to balance GPU memory utilization and gradient stability. |
| **Total Epochs** | 200 | Ensured loss convergence without overfitting. |
| **Initial Learning Rate** | $1.0 \times 10^{-4}$ | Standard Adam optimizer starting learning rate. |
| **Learning Rate Decay** | $5.0 \times 10^{-5}$ per step | Decays learning rate via $\eta_t = \eta_0 / (1 + \text{decay} \times t)$. |
| **Style Weight ($\lambda_s$)**| 10.0 | Balances stylization intensity against structural content loss. |
| **Content Weight** | 1.0 | Baseline weighting for content feature matching. |
| **Training Resolution** | 512px resize, 256px random crop | Reduces compute overhead while preserving multi-scale texture details. |
| **Checkpoint Output** | `decoder_final.pth` (14.0 MB) | Serialized state dictionary of the trained decoder network. |

---

## 6.4 Flask Web Application and UI Implementation
The web layer provides an intuitive interface for running style transfer without command-line tools:
- **Form Handling:** Uses `Flask-WTF` with `FileField`, hidden fields for preset style paths, and a `FloatField` for $\alpha$ transfer strength.
- **Preset Gallery:** Offers 16 curated artwork styles (including *Starry Night*, *Picasso Seated Nude*, *La Muse*, *Mondrian*, *Woman in Peasant Dress*, and *Scene de Rue*). Clicking a preset thumbnail populates the hidden style field instantly.
- **Inference Execution:** The `/` route validates image uploads, securely saves files to `static/uploads/`, checks image integrity, loads tensors, runs inference via `run_style_transfer()`, and applies text preservation where applicable.
- **Stateless Storage:** Temporary files are saved locally with timestamped filenames, avoiding external database dependencies.

---

## 6.5 Module-wise Design

### Table 6.2: Module-wise Specifications, Processing, and Dependencies
| Module | Purpose | Inputs | Processing | Outputs | Dependencies |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Feature Extraction & Statistical Alignment Module** (`models.py`, `utils.py`) | Extracts deep semantic feature representations from content and style images and aligns channel-wise statistics in real time. | Content tensor $I_c$, Style tensor $I_s$ ($3 \times 512 \times 512$). | Passes inputs through frozen VGG-19 truncated at `relu4_1`; computes spatial mean $\mu$ and std $\sigma$ via `calc_mean_std()`; applies AdaIN normalization $\sigma(f_s) \frac{f_c - \mu(f_c)}{\sigma(f_c)} + \mu(f_s)$. | Normalized bottleneck feature tensor $t \in \mathbb{R}^{512 \times 64 \times 64}$. | `torch`, `torch.nn`, `torchvision.transforms`, `vgg_normalised.pth`. |
| **2. Latent Interpolation & Image Reconstruction Decoder** (`models.py`, `train.py`) | Interpolates feature maps based on user transfer strength and inverts latent representations back into RGB pixel space. | Normalized feature $t$, unstylized content feature $f_c$, user alpha $\alpha \in [0.0, 1.0]$. | Computes linear interpolation $T = \alpha \cdot t + (1 - \alpha) \cdot f_c$; routes tensor $T$ through 4-stage symmetrical convolutional decoder using reflection padding and nearest upsampling. | Stylized RGB image tensor / PIL Image ($I_{\text{stylized}} \in \mathbb{R}^{3 \times 512 \times 512}$). | `torch`, `torch.nn`, `decoder_final.pth`. |
| **3. Dynamic Text Detection & Feathered Blending Module** (`app.py`) | Automatically localizes text regions on content images and blends them seamlessly over stylized output to preserve readability. | Original content PIL image, stylized image, transfer strength $\alpha$. | Runs EasyOCR CRAFT text detector; expands boxes with 4px padding; applies Gaussian blur ($r=3$); modulates mask weight $w_{\text{text}}$ based on $\alpha$; composites unstylized text over stylized base. | Legible, text-preserved composite PIL Image ($I_{\text{out}}$). | `easyocr`, `PIL.Image`, `PIL.ImageFilter`, `PIL.ImageDraw`, `numpy`. |
| **4. Full-Stack Web Controller & Interface Module** (`app.py`, `index.html`) | Handles HTTP requests, provides drag-and-drop UI, validates input files, routes model inference, and serves downloads. | Multipart HTTP POST requests, uploaded image files, preset selections, slider value $\alpha$. | Validates file extensions and MIME types with WTForms; sanitizes filenames with Werkzeug; selects CUDA or CPU fallback; coordinates pipeline inference; renders output view. | Rendered HTML view, client feedback, generated image files in `static/uploads/`. | `flask`, `flask_wtf`, `wtforms`, `werkzeug.utils`. |

---

### Module Sum-Up
The system is built on a high-cohesion, low-coupling modular architecture. The **Feature Extraction & Alignment Module** and the **Decoder Reconstruction Module** operate strictly on PyTorch tensors, allowing them to function independently of the web framework. The **Dynamic Text Preservation Module** acts as an intelligent computer vision filter that can be toggled without altering network weights. Finally, the **Web Controller Module** acts as an orchestrator, decoupling user interaction and file management from deep learning computation. This modular separation ensures high maintainability, easy testing, and smooth execution across both GPU and CPU environments.

---
---

# CHAPTER 7
# RESULTS AND DISCUSSION

## 7.1 Visual Quality and Arbitrary Style Adaptation
The trained AdaIN decoder successfully transfers styles across a wide range of artistic inputs. Unlike single-style networks, the model generalizes to styles it was not exposed to during training.

- **Line Drawings & Sketches:** When using sketch references (e.g., `sketch.png`), the network extracts structural linework and light hatching while preserving the underlying object contours.
- **Cubist & Expressionist Art:** When using works by Picasso (e.g., `picasso_seated_nude_hr.jpg`), the model applies geometric color blocking, rich earthy pigments, and angular contours to the content image.
- **Impressionism & Modernism:** Testing with Mondrian presets transfers structured primary-color palettes and rectilinear grid textures without compromising core object geometry.

---

## 7.2 Style Intensity Control ($\alpha$ Interpolation Results)
Adjusting the style intensity slider ($\alpha \in [0.0, 1.0]$) changes the feature representation smoothly:
- $\alpha = 0.0$: The output reconstructs the original content image with minimal stylistic influence.
- $\alpha = 0.25$: Adds subtle color grading and light texturing while preserving original photographic detail.
- $\alpha = 0.50$: Produces a balanced output where content details and artistic brushwork are equally prominent.
- $\alpha = 0.75$: The artistic style becomes dominant, with painterly textures applied across major surfaces.
- $\alpha = 1.0$: Full stylization with maximum artistic texture, brushstrokes, and palette shifts.

Linear feature-space interpolation avoids the double-exposure or ghosting artifacts typically seen with simple pixel-level image blending.

---

## 7.3 Text Legibility Preservation Evaluation
Comparative evaluations show that standard AdaIN blurs and distorts textual elements in posters, book covers, and street signs into textured paint strokes.

By contrast, the integrated EasyOCR text-preservation module locates text bounding boxes and applies feathered Gaussian blending:
- At $\alpha \le 0.5$, text retains crisp edges and high contrast, remaining clear and readable.
- At $\alpha > 0.5$, text takes on subtle palette hues from the style image while remaining legible, blending naturally into the stylized background.
- Feathered Gaussian borders ($\text{radius} = 3$) eliminate harsh masking seams, producing a cohesive visual result.

---

## 7.4 Latency and Inference Benchmarks (CPU vs. GPU)
To evaluate responsiveness, inference was benchmarked across standard test images at $512 \times 512$ resolution.

### Table 7.1: Quantitative Inference Latency Across Processing Units
| Processing Stage | NVIDIA Dedicated GPU (CUDA) | Multi-Core CPU Fallback | Performance Impact |
| :--- | :--- | :--- | :--- |
| **VGG-19 Feature Encoding** | ~0.12 seconds | ~0.85 seconds | GPU yields ~7× faster feature extraction. |
| **AdaIN Alignment & Interpolation** | ~0.01 seconds | ~0.04 seconds | Lightweight tensor operations in both modes. |
| **Decoder Image Reconstruction** | ~0.22 seconds | ~1.65 seconds | GPU yields ~7.5× faster convolutional decoding. |
| **EasyOCR Text Detection** | ~0.25 seconds | ~0.75 seconds | CRAFT inference runs efficiently on both platforms. |
| **Total End-to-End Latency** | **~0.60 seconds** | **~3.29 seconds** | Real-time on GPU; acceptable on CPU fallback. |

The feed-forward AdaIN network achieves sub-second inference on modern GPUs and runs in ~3 seconds on standard CPUs, making it practical for local and educational deployments.

---

## 7.5 Verification and Functional Test Cases
The application was validated using automated test scripts (`test_app.py`, `test_full.py`) and manual test cases covering core user interactions.

### Table 7.2: Functional Integration Test Suite Summary
| Test Case ID | Test Description | Input Condition | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TC-01** | Root Route Availability | HTTP `GET /` | Returns HTTP 200 and loads interface. | HTTP 200 returned; UI rendered cleanly. | Pass |
| **TC-02** | Valid Image Submission | Content JPG + Style preset + $\alpha=0.8$ | Generates stylized image in `static/uploads/`. | Output image rendered and displayed. | Pass |
| **TC-03** | Unsupported File Type | Content file `.txt` uploaded | Form rejection with error message. | Handled with informative validation error. | Pass |
| **TC-04** | Missing Style Selection | Content uploaded without style | Prompt user to pick a style. | Handled with validation warning. | Pass |
| **TC-05** | Extreme $\alpha$ Values | $\alpha = 0.0$ and $\alpha = 1.0$ | Stable image generation at boundaries. | Reconstructs content at 0.0; full style at 1.0. | Pass |
| **TC-06** | Text-Heavy Image Upload | Image with English text headline | Detects text and applies feathered mask. | Text remains crisp and legible in output. | Pass |

---
---

# CHAPTER 8
# LIMITATIONS AND FUTURE SCOPE

## 8.1 Current Technical Limitations
While the system performs well across diverse images, a few technical limitations remain:
1. **Resolution vs. Memory Trade-Off:** Inference defaults to 512px along the primary dimension to maintain responsive speeds and prevent CUDA out-of-memory errors on entry-level GPUs. Processing ultra-high-resolution images (4K or higher) would require tiling strategies.
2. **Text Detection Edge Cases:** EasyOCR detects standard horizontal and polygonal text reliably, but can occasionally miss heavily stylized, cursive, or low-contrast lettering.
3. **Absence of Temporal Smoothing:** The system is designed for static images. Applying the model frame-by-frame to video produces noticeable inter-frame flicker because statistical feature distributions can vary between frames.

---

## 8.2 Recommended Future Work
1. **Video Style Transfer with Temporal Loss:** Incorporating optical flow or temporal consistency loss terms would enable flicker-free video stylization for short animations and recorded clips.
2. **Interactive Spatial Style Brushing:** Expanding the web UI with a canvas-based brush tool would allow users to apply different artistic styles to specific regions of an image.
3. **Multilingual and Handwritten OCR:** Expanding OCR language models and adding specialized handwriting detection would improve text preservation across a broader range of visual materials.
4. **Mobile Deployment via ONNX / TensorRT:** Quantizing the trained decoder network to INT8 via ONNX Runtime or TensorRT would allow efficient deployment on mobile devices and edge hardware.

---
---

# CHAPTER 9
# CONCLUSION

## 9.1 Summary of Contributions
This project designed, implemented, and validated **Art Fusion (CanvasFlow)**, an end-to-end arbitrary neural style transfer system built on Adaptive Instance Normalization in PyTorch and deployed as a responsive Flask web application.

Key contributions include:
- Implementing the AdaIN statistical alignment layer, which eliminates per-image optimization and transfers arbitrary, unseen artistic styles in real time.
- Training a 4-stage convolutional decoder on MS-COCO 2017 and WikiArt datasets using a composite perceptual loss function across multiple VGG-19 layers.
- Providing continuous style intensity control ($\alpha \in [0.0, 1.0]$) through linear feature-space interpolation.
- Engineering an automated text-preservation pipeline using EasyOCR and dynamic alpha-weighted Gaussian feathered blending, addressing the common problem of typographic distortion in NST.
- Building a user-friendly web interface with 16 artistic presets and CPU/GPU auto-detection.

---

## 9.2 Key Viva Defense Takeaways

1. **Why AdaIN instead of Gatys' optimization method?**  
   Gatys' method relies on iterative gradient descent over image pixels, taking minutes per image. AdaIN aligns channel statistics in a single feed-forward pass, reducing generation time to fractions of a second.

2. **Why AdaIN instead of Johnson's feed-forward network?**  
   Johnson's network is tied to a single style learned during training. AdaIN dynamically adjusts feature statistics at runtime, transferring arbitrary unseen styles using a single trained decoder.

3. **What is the mathematical role of AdaIN?**  
   It normalizes content features to zero mean and unit variance, then scales and shifts them using the standard deviation and mean of the style features:  
   $$\text{AdaIN}(x, y) = \sigma(y) \left( \frac{x - \mu(x)}{\sigma(x)} \right) + \mu(y)$$

4. **Why are VGG-19 encoder weights frozen during training?**  
   The encoder acts solely as a feature extractor. Freezing its weights ensures that feature representations remain consistent while the decoder learns to invert aligned feature maps back into realistic RGB images.

5. **How is text legibility preserved without manual masks?**  
   EasyOCR locates text regions in the content image. A Gaussian-blurred feathered mask is generated, and its blending strength is scaled dynamically based on the style slider ($\alpha$), keeping lettering crisp and readable.

---
---

# REFERENCES

1. Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). *Image style transfer using convolutional neural networks*. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2414–2423.
2. Huang, X., & Belongie, S. (2017). *Arbitrary style transfer in real-time with adaptive instance normalization*. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 1501–1510.
3. Johnson, J., Alahi, A., & Fei-Fei, L. (2016). *Perceptual losses for real-time style transfer and super-resolution*. In European Conference on Computer Vision (ECCV), Springer, pp. 694–711.
4. Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016). *Instance normalization: The missing ingredient for fast stylization*. arXiv preprint arXiv:1607.08022.
5. Li, Y., Fang, C., Yang, J., Wang, Z., Lu, X., & Yang, M. H. (2017). *Universal style transfer via feature transforms*. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 30, pp. 386–396.
6. Baek, Y., Lee, B., Han, D., Yun, S., & Lee, H. (2019). *Character region awareness for text detection*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9365–9374.
7. Simonyan, K., & Zisserman, A. (2014). *Very deep convolutional networks for large-scale image recognition*. arXiv preprint arXiv:1409.1556.
8. Lin, T. Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). *Microsoft COCO: Common objects in context*. In European Conference on Computer Vision (ECCV), Springer, pp. 740–755.

---
---

# APPENDICES

### Appendix A: Core AdaIN and Statistical Alignment Code
```python
import torch

def calc_mean_std(feat, eps=1e-5):
    """Calculate channel-wise mean and standard deviation across spatial dimensions."""
    size = feat.size()
    assert len(size) == 4
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    """Normalize content activations and shift/scale with style statistics."""
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)
```

### Appendix B: Dynamic Text Blending Logic
```python
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

def create_text_mask(content_pil, target_size, ocr_reader, alpha=1.0):
    """Detect text bounding boxes and generate an alpha-responsive feathered mask."""
    orig_w, orig_h = content_pil.size
    target_w, target_h = target_size
    scale_x, scale_y = target_w / orig_w, target_h / orig_h

    np_img = np.array(content_pil.convert('RGB'))
    try:
        horizontal_list, free_list = ocr_reader.detect(np_img)
    except Exception:
        return None

    boxes = horizontal_list[0] if horizontal_list else []
    polys = free_list[0] if free_list else []
    if not boxes and not polys:
        return None

    mask = Image.new('L', target_size, 0)
    draw = ImageDraw.Draw(mask)
    pad = 4

    for box in boxes:
        x_min = max(0, int(box[0] * scale_x) - pad)
        x_max = min(target_w, int(box[1] * scale_x) + pad)
        y_min = max(0, int(box[2] * scale_y) - pad)
        y_max = min(target_h, int(box[3] * scale_y) + pad)
        draw.rectangle([x_min, y_min, x_max, y_max], fill=255)

    for poly in polys:
        scaled_poly = [(int(pt[0] * scale_x), int(pt[1] * scale_y)) for pt in poly]
        draw.polygon(scaled_poly, fill=255)

    feathered = mask.filter(ImageFilter.GaussianBlur(radius=3))
    alpha_clamped = max(0.0, min(1.0, float(alpha)))

    # Compute dynamic preservation weight
    if alpha_clamped <= 0.5:
        text_weight = 1.0 - (alpha_clamped / 0.5) * 0.20
    else:
        progress = (alpha_clamped - 0.5) / 0.5
        text_weight = 0.80 - progress * 0.70

    return feathered.point(lambda p: int(p * text_weight))
```
