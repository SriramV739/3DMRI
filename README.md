🥼 Surge AI

An Edge-Native, Software-Defined Medical Device (SDMD) for Intra-Operative Augmented Reality & AI Consultation.

This project is a localized software suite designed to enhance surgical precision. It ingests patient MRI/CT data to generate 3D anatomical reference models and processes live endoscopic video to provide real-time semantic highlighting. Additionally, it features a voice-activated "Surgical Pause" Copilot powered by Vision-Language Models (VLMs) for on-demand clinical reasoning.
🧠 System Architecture

This system operates across four distinct phases, transitioning from pre-operative planning to real-time intra-operative execution.
Phase 1: Pre-Op 3D Geometry Generation (Offline)

Extracts mathematical geometry from static patient scans to provide the surgeon with a rotatable, 3D map of the patient's anatomy prior to incision.

    Input: Raw volumetric medical scans (DICOM / NIfTI).

    AI Engine: NVIDIA MONAI (Swin UNETR) isolates target organs/anomalies in the 3D voxel space.

    Geometry Engine: Marching Cubes (scikit-image) traces the boundary of the segmented mask to create a polygonal mesh.

    Optimization: Trimesh applies Laplacian smoothing and decimation, exporting a lightweight .obj file.

Phase 2: Live Intra-Op Augmented Reality (The Core)

The high-performance execution loop. It provides persistent, zero-latency semantic highlighting of critical structures.

    Orchestrator: NVIDIA Holoscan SDK pulls video directly into GPU VRAM (bypassing the CPU) via VideoStreamReplayOp.

    Tracking Brain: Meta SAM 2.1 operates natively within a Holoscan Inference Operator, using temporal memory to output 60+ FPS augmented reality masks on moving tissues.

Phase 3: The "Surgical Pause" VLM Copilot

An interactive intelligence layer allowing the surgeon to ask complex, multimodal questions about the visible anatomy hands-free.

    Voice Trigger: faster-whisper constantly monitors audio locally. Saying "System, pause" freezes the Holoscan video feed.

    Clinical Reasoning: The frozen frame and transcribed question are passed to NVIDIA Cosmos Reason (via NIM/API).

    Grounded Output: The VLM returns clinical text analyzing the tissue and generates precise spatial coordinates/masks of the queried anomaly.

Phase 4: Holoviz Compositing UI

The final visualization environment leveraging GPU-native rendering.

    Left Viewport (Map): Interactive 3D render of the .obj generated in Phase 1.

    Right Viewport (Live): Composited live endoscopic video with SAM 2.1's colored tracking masks.

    Pause State: Overlays the VLM’s diagnostic mask and text response in a high-contrast UI block when a consultation is triggered.
