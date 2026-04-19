# Integration Steps: Phase 1 (CT) & Phase 2 (Surgery)

This document tracks the concrete steps required to merge the CT Vision Pro application branch with the Intra-Op Surgery AR branch into a unified repository, while strictly respecting their different deployment platforms (Vision Pro vs Computer/OR Monitor).

## 1. Git Merging
- [ ] Ensure you are on the `CT_stuff` (or `main`) branch.
- [ ] Run `git fetch --all`.
- [ ] Execute `git merge origin/surgery --no-edit`. 
- [ ] Verify that the `surgery/` folder now exists at the repository root immediately next to the `CT/` folder.

## 2. Global Configuration Unification
- [ ] Move `/Users/asteyalaxmanan/3DMRI/CT/.env` to `/Users/asteyalaxmanan/3DMRI/.env` (repo root).
- [ ] Update `CT/backend/settings.py` to point its `load_dotenv` call to the new root `.env` file.
- [ ] Update `surgery/config/app_config.yaml` or relevant code to read AI endpoint tokens from the new root `.env` file.

## 3. Platform & Environment Integrity
- [ ] **CT App Verification**: Leave `CT/requirements.txt` alone. The CT pipeline (FastAPI + VisionOS generation) retains its current `CT/.venv`.
- [ ] **Surgery App Verification**: Leave `surgery/requirements.txt` alone. The Surgery pipeline (Holoscan + YOLO + MedSAM2) stays isolated for Linux/NVIDIA execution and remains firmly targeted for PC desktop viewing.
- [ ] Update `/Users/asteyalaxmanan/3DMRI/.gitignore` to ignore the new `surgery/` generated cache folders (e.g. `surgery/.venv` and `.holoscan`).

## 4. Final Platform Check
- [ ] Boot the CT backend and ensure the Vision Pro simulator loads the 3D meshes flawlessly.
- [ ] Ensure no Surgery or Live-AR elements were accidentally exposed or tied to the VisionOS payload generation.
