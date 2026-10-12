# Scientific Paper LaTeX Publication Package

This directory contains the complete LaTeX source package for the scientific journal manuscript:

> **Title:** *Physics-Informed Deep Learning for Exoplanet Transit Detection: Discovery of a Sub-stellar Candidate Orbiting the Unvetted K-dwarf TIC 200668668*  
> **Author:** Nicu Drumea  
> **Target Journals:** The Astrophysical Journal (ApJ), Monthly Notices of the Royal Astronomical Society (MNRAS), The Astronomical Journal (AJ), or arXiv (astro-ph.EP).

---

## Files in this Directory

- [`manuscript.tex`](manuscript.tex): Full LaTeX source code of the paper with abstract, mathematical derivations, tables, and discussions.
- [`references.bib`](references.bib): BibTeX bibliography database with all literature citations (Mandel & Agol 2002, Kovács et al. 2002, Ricker et al. 2015, Shallue & Vanderburg 2018, Pearson et al. 2018, Raissi et al. 2019, Guerrero et al. 2021, etc.).
- [`README.md`](README.md): Instructions for local compilation and Overleaf deployment.

---

## Option 1: Import into Overleaf (Recommended for Fast Collaboration)

1. Open [https://www.overleaf.com](https://www.overleaf.com) and log in.
2. Click **New Project** -> **Upload Project**.
3. Select and upload the `paper/` folder (or a `.zip` containing `manuscript.tex` and `references.bib`).
4. Overleaf will automatically compile the document into a publication-ready PDF.

---

## Option 2: Local Compilation with LaTeX

If you have TeX Live or MiKTeX installed locally:

```bash
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

This generates `manuscript.pdf`.

---

## Key Data and Discovery Artifacts Associated with this Paper

- **Calibrated Model Checkpoint:** `artifacts/pinn-transnet-v1.pt`
- **Candidate Diagnostic Report:** `artifacts/tic-200668668-candidate-report.html`
- **Unvetted Discovery Batch Catalog:** `artifacts/discovery_unvetted_batch1.csv`
- **Permanent Open Science DOI:** `10.5281/zenodo.14889211`
- **Source Code Repository:** [https://github.com/Nicu2806/astromap-ai](https://github.com/Nicu2806/astromap-ai)
