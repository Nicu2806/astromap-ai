{
  "project_metadata": {
    "title": "AstroMap-AI: Physics-Informed Interactive Exoplanet Detection System",
    "project_type": "Master Thesis / Applied Scientific Software",
    "domain": "Astrophysics, Deep Learning, Spatial Web Systems",
    "version": "1.0.0",
    "target_hardware": "NVIDIA GPU (CUDA >= 12.0) for training, standard web browser for visualization"
  },
  "executive_summary": {
    "objective": "Develop an end-to-end scientific platform that ingests raw photometric light curves from NASA TESS/Kepler, applies an innovative Physics-Informed Wavelet-Attention Neural Network to detect exoplanetary transits, and visualizes candidate systems on an interactive celestial sphere.",
    "core_innovation": "Bypassing rigid pre-folding period searches (BLS) by combining 1D Continuous Wavelet Transform (CWT) time-frequency representations with a Physics-Informed Loss Function based on the Mandel-Agol analytic transit model to suppress instrumental false positives and detect Transit Timing Variations (TTV)."
  },
  "agent_instructions": {
    "role": "Lead Research Engineer and Full-Stack Scientific Software Architect",
    "workflow_rule": "Execute sequentially from Phase 1 through Phase 5. Ensure reproducible data pipelines, deterministic training seeds, and modular API endpoints.",
    "coding_standards": {
      "python": "Python 3.11+, typed hints (PEP 484), Black formatter, PyTorch >= 2.2",
      "frontend": "TypeScript, React 18+, TailwindCSS, Vite",
      "documentation": "Docstrings in Google style, clear mathematical equations referenced in comments"
    }
  },
  "modules": [
    {
      "module_id": "MOD_01_INGESTION_PREPROCESSING",
      "name": "Data Ingestion and Astro-Photometric Pipeline",
      "dependencies": ["lightkurve", "astropy", "numpy", "scipy", "pandas"],
      "tasks": [
        {
          "step": 1,
          "action": "Query and download target pixel files / light curves from NASA MAST archive using lightkurve API.",
          "sources": [
            "TESS Sector 1-70 SPOC light curves",
            "Kepler Q1-Q17 calibrated light curves",
            "NASA Exoplanet Archive TOI and TCE labeled catalogs (Confirmed, Candidate, False Positive)"
          ]
        },
        {
          "step": 2,
          "action": "Implement photometric cleaning pipeline.",
          "operations": [
            "Mask NaN values and flag-tagged instrumental anomalies (momentum dumps, cosmic rays).",
            "Remove stellar variability and low-frequency trends using Savitzky-Golay filtering and Gaussian Processes.",
            "Normalize median flux to 1.0."
          ]
        },
        {
          "step": 3,
          "action": "Generate Dual-Domain Representations.",
          "details": "Compute 1D Continuous Wavelet Transform (CWT) using a Morlet or Complex Ricker wavelet to extract localized time-frequency scalograms, capturing sharp U-shaped transit signatures vs. V-shaped stellar binary dips."
        }
      ]
    },
    {
      "module_id": "MOD_02_AI_CORE",
      "name": "Physics-Informed Neural Transit Detector (PINN-TransNet)",
      "dependencies": ["torch", "torchvision", "batman-package", "scikit-learn"],
      "architecture": {
        "encoder": "1D-CNN + Temporal Convolutional Network (TCN) multi-scale feature extractor.",
        "attention_layer": "Multi-Head Temporal Self-Attention to capture non-local periodic dependencies (supporting TTVs).",
        "dual_head_output": {
          "classification_head": "Sigmoid output: P(Exoplanet Transit | Signal).",
          "parameter_regression_head": "Estimates transit parameters: Depth (delta), Duration (T_dur), Period (P)."
        }
      },
      "loss_function": {
        "mathematical_formula": "L_total = L_BCE(y_pred, y_true) + lambda_phys * L_physics",
        "physics_constraint": "Penalize transit depth and duration values that violate the Mandel-Agol (2002) physical equations given known stellar radius (R_star) and surface gravity (log g) retrieved from the Gaia DR3 cross-match."
      },
      "benchmarking_metrics": [
        "ROC-AUC",
        "Precision-Recall AUC (critical for imbalanced data)",
        "False Positive Rate at 95% Recall",
        "Comparison against classic Box-fitting Least Squares (BLS) + Random Forest baseline"
      ]
    },
    {
      "module_id": "MOD_03_BACKEND_API",
      "name": "Inference and Data Serving Backend",
      "stack": ["FastAPI", "Uvicorn", "Celery", "Redis", "SQLite/PostgreSQL"],
      "endpoints": [
        {
          "route": "/api/v1/targets/search",
          "method": "GET",
          "params": ["ra", "dec", "radius_deg"],
          "description": "Returns list of stars within coordinates with metadata and pre-evaluated prediction scores."
        },
        {
          "route": "/api/v1/analyze/tic/{tic_id}",
          "method": "POST",
          "description": "Triggers live download, pre-processing, wavelet transform, and model inference for a given TESS Input Catalog (TIC) ID."
        },
        {
          "route": "/api/v1/lightcurve/{tic_id}",
          "method": "GET",
          "description": "Returns raw, flattened, and folded photometric series as JSON for charting."
        }
      ]
    },
    {
      "module_id": "MOD_04_CELESTIAL_UI",
      "name": "Interactive Sky Map and Diagnostic Dashboard",
      "stack": ["React", "Aladin Lite v3 API", "Plotly.js / Chart.js", "TailwindCSS"],
      "features": [
        {
          "name": "Hierarchical Sky Viewer",
          "implementation": "Embed CDS Aladin Lite v3 rendering HiPS optical/infrared sky surveys. Project target stars as interactive markers colored by predicted probability (Green: Exoplanet, Red: False Positive/Binary, Grey: Unchecked)."
        },
        {
          "name": "Target Inspection Panel",
          "implementation": "Split-view pane showing interactive light curve with zoom/pan, phase-folded transit profile, scalogram heatmaps, and physical parameter estimations (planetary radius, orbital period)."
        },
        {
          "name": "Filtering and Query Controls",
          "implementation": "Filter targets by Right Ascension (RA), Declination (Dec), transit SNR, magnitude, and spectral type."
        }
      ]
    },
    {
      "module_id": "MOD_05_EVALUATION_REPORTING",
      "name": "Scientific Validation and Master Thesis Export",
      "deliverables": [
        "Ablation study results (Model with Physics Loss vs. Standard Cross-Entropy).",
        "Automated PDF candidate validation report generation for newly flagged TOIs.",
        "Complete reproducible Jupyter Notebooks demonstrating pipeline flow from raw FITS to candidate validation."
      ]
    }
  ],
  "implementation_roadmap": {
    "milestone_1": "Setup pipeline, download 10,000 labeled TESS/Kepler curves, implement preprocessing and CWT.",
    "milestone_2": "Build and train PINN-TransNet model; benchmark against classic BLS.",
    "milestone_3": "Develop FastAPI microservice with asynchronous task queuing for live inference.",
    "milestone_4": "Integrate Aladin Lite UI with backend APIs and diagnostic light curve plots.",
    "milestone_5": "Validate system on unconfirmed TESS candidates and compile thesis results."
  }
}