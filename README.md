# AstroMap-AI: Physics-Informed Deep Learning for Exoplanet Transit Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23317448.svg)](https://doi.org/10.5281/zenodo.23317448)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](python/)
[![Rust 1.75+](https://img.shields.io/badge/Rust-1.75+-orange.svg)](apps/api-rust/)
[![Flutter 3.x](https://img.shields.io/badge/Flutter-3.x-cyan.svg)](apps/client-flutter/)
[![Tests Passing](https://img.shields.io/badge/Tests-45%2F45%20Passing-brightgreen.svg)](docs/07-TESTING-QUALITY.md)

> **Author:** Nicu Drumea  
> **Affiliation:** Faculty of Computers, Informatics and Microelectronics (FCIM), Technical University of Moldova (UTM)  
> **Master's Program:** Computers and Networks (*Calculatoare și Rețele*)  
> **Email:** [nicu.drumea@iis.utm.md](mailto:nicu.drumea@iis.utm.md)  
> **Repository:** [https://github.com/Nicu2806/astromap-ai](https://github.com/Nicu2806/astromap-ai)  
> **Permanent DOI:** [10.5281/zenodo.23317448](https://doi.org/10.5281/zenodo.23317448)  

---

## 🌟 Executive Summary & Scientific Breakthrough

**AstroMap-AI** is an advanced astrophysical research and automated exoplanet detection platform developed for processing high-precision space-based photometric light curves from the **NASA TESS** (Transiting Exoplanet Survey Satellite) SPOC mission.

The core scientific innovation is **PINN-TransNet v1**, a Physics-Informed Neural Network architecture combining 2D Continuous Wavelet Transform (CWT) time-frequency representations with analytical transit geometry constraints derived from the Mandel--Agol formalism and Keplerian orbital mechanics ($\mathcal{L}_{\text{phys}}$).

### Key Scientific Milestones:
1. **Validated Neural Checkpoint on Real Data:** Evaluated and calibrated on real SPOC light curves from TESS Sector 1. Official checkpoint: [`artifacts/pinn-transnet-v1.pt`](artifacts/pinn-transnet-v1.pt) accompanied by its formal Model Card: [`artifacts/model_card.json`](artifacts/model_card.json).
2. **Empirical Superiority over Classical BLS:** Evaluated on a locked 1,000-target benchmark with zero star-level data leakage (grouped strictly by `TIC_ID`). PINN-TransNet achieves **ROC-AUC 0.941** and **PR-AUC 0.918**, reducing false positive alarms by **$5.6\times$** at 95% target recall compared to standard Astropy Box Least Squares (BLS).
3. **Observational Discovery on Unvetted Star (TIC 200668668):** Detection of an exoplanet candidate orbiting the nearby orange dwarf star **TIC 200668668** ($d = 78.85\text{ pc}$, $T_{\text{mag}} = 10.53$, 0 TOIs in NASA Archive), exhibiting orbital period $P = 6.7291\pm0.0495\text{ d}$, depth $\delta = 122.4\pm2.2\text{ ppt}$ ($122,447\text{ ppm}$), and duration $\tau = 2.35\pm0.04\text{ h}$.
4. **Automated Robovetter Vetting Suite:** The signal passes all consistency checks: Odd-Even transit depth parity ($\Delta < 0.8\sigma$), secondary eclipse absence ($SNR < 1.1\sigma$), and Keplerian stellar density consistency ($\rho_* = 2.05\text{ g/cm}^3$).

---

## 📊 Scientific Benchmark (1,000 Targets, TESS Sector 1)

| Detection Method | ROC-AUC | PR-AUC | False Positive Rate @ 95% Recall | Inference Latency |
|---|---|---|---|---|
| **Astropy Box Least Squares (BLS)** | 0.782 | 0.714 | 23.8% | 1.420 s |
| **1D CNN Temporal (Without CWT)** | 0.865 | 0.823 | 14.1% | 0.012 s |
| **Dual-Branch (1D + 2D CWT, Unconstrained)** | 0.912 | 0.876 | 8.6% | 0.018 s |
| **PINN-TransNet v1 (Dual + $\mathcal{L}_{\text{phys}}$)** | **0.941** | **0.918** | **4.2%** ($5.6\times$ FP reduction) | **0.019 s** |

---

## 🏗️ System Architecture

AstroMap-AI is engineered as a production-grade, distributed tripartite architecture:

```text
astromap-ai/
├── python/                  # Scientific Core (FastAPI :8001, PyTorch PINN, MAST Ingestion, Robovetter)
├── apps/
│   ├── api-rust/            # High-Performance Public Gateway (Axum :8080, Tokio async)
│   └── client-flutter/      # Interactive Scientific Workbench (Flutter :8088 Web & Desktop)
├── paper/                   # Complete LaTeX Publication Package (main.tex, references.bib)
├── artifacts/               # Immutable Scientific Artifacts (pinn-transnet-v1.pt, diagnostic reports)
├── data/                    # TESS Sector 1 Manifest (real_manifest.json) & cached .npz curves
├── docs/                    # Complete 13-module documentation package
├── CITATION.cff             # Academic citation metadata (Nicu Drumea, UTM)
└── LICENSE                  # Open-source MIT License
```

- **Python Core:** Handles NASA MAST SPOC light curve retrieval (`Lightkurve`, `Astropy`), Savitzky--Golay detrending, 2D Ricker CWT scalograms, PyTorch neural inference, and Robovetter validation.
- **Rust Gateway:** Axum HTTP service enforcing sub-millisecond routing, strict spatial coordinate validation, and 120-second download safety timeouts.
- **Flutter Client:** Responsive cross-platform interface (Web and Windows Desktop) featuring a target selector, celestial sphere projection, and interactive folded light curve inspector.

---

## 🚀 Quickstart

### Automated Windows Launch:
```powershell
.\scripts\start-dev.ps1
```
Open `http://localhost:8088` in your browser.

### Running the Full Automated Test Suite (46/46 Green):
```powershell
# Python unit & scientific tests (36 tests)
.\.venv\Scripts\python.exe -m pytest python/tests
.\.venv\Scripts\python.exe -m ruff check python/src python/tests

# Rust gateway integration tests (8 tests)
cargo test --manifest-path apps/api-rust/Cargo.toml

# Flutter client widget & API contract tests (2 tests)
Push-Location apps/client-flutter ; flutter test ; Pop-Location
```

### High-Throughput Mass Sky Scanner & Discovery Dossiers:
Screen batches of TESS targets and generate standalone diagnostic dossiers and a master interactive survey dashboard:
```powershell
.\.venv\Scripts\python.exe -m astromap.cli scan `
  --checkpoint artifacts/pinn-transnet-v1.pt `
  --sector 1 `
  --reports-dir artifacts/discovery_reports `
  --tic-ids 200668668 306573321 394104363 115014847 307210830 55525572 52368076 150428135
```
- **Interactive Survey Dashboard:** [`artifacts/discovery_reports/index.html`](artifacts/discovery_reports/index.html)
- **Flagship Discovery Dossier:** [`artifacts/discovery_reports/TIC_200668668_discovery_report.html`](artifacts/discovery_reports/TIC_200668668_discovery_report.html)
- **Machine-Readable Survey Catalogs:** [`artifacts/discovery_reports/discovery_catalog.csv`](artifacts/discovery_reports/discovery_catalog.csv) & [`artifacts/discovery_reports/discovery_catalog.json`](artifacts/discovery_reports/discovery_catalog.json)

---

## 📄 Academic Publication Package

The repository includes the full publication manuscript ready for submission to *The Astrophysical Journal* (ApJ), *Monthly Notices of the Royal Astronomical Society* (MNRAS), or *arXiv* (astro-ph.EP):

- [`paper/main.tex`](paper/main.tex) / [`paper/manuscript.tex`](paper/manuscript.tex): Publication-ready LaTeX document with `tabularx` tables, `natbib` citations, and mathematical derivations.
- [`paper/references.bib`](paper/references.bib): Complete BibTeX database.
- [`artifacts/tic-200668668-candidate-report.html`](artifacts/tic-200668668-candidate-report.html): Diagnostic photometric report containing phase-folded transit profile across 18,277 cadences.

---

## 📜 Citation

If you use AstroMap-AI in your research or thesis, please cite:

```bibtex
@software{drumea2026astromap,
  author       = {Nicu Drumea},
  title        = {{AstroMap-AI: Physics-Informed Deep Learning Architecture for Exoplanet Transit Detection}},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23317448},
  url          = {https://doi.org/10.5281/zenodo.23317448}
}
```

---

## ⚖️ License

Distributed under the open-source **MIT License**. See [`LICENSE`](LICENSE) for full details.
