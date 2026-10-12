# Ghid de Publicare Zenodo & Open Science cu DOI Permanent

> **Autor:** Nicu Drumea  
> **Proiect:** AstroMap-AI  
> **Obiectiv:** Asignarea unui identificator digital permanent (**DOI - Digital Object Identifier**) prin Zenodo (CERN) conectat la repository-ul GitHub, asigurând citabilitatea științifică a codului, a checkpoint-ului `PINN-TransNet v1` și a datelor descoperirii candidatului **TIC 200668668**.

---

## 1. De ce este Esențial Zenodo pentru Publicarea Științifică?

Revistele astronomice de top (AAS Journals / *The Astrophysical Journal*, *Monthly Notices of the Royal Astronomical Society*, *Astronomy & Astrophysics*) solicită respectarea principiilor **FAIR** (*Findable, Accessible, Interoperable, Reusable*):
1. **Atribuire de autorat incontestabilă:** DOI-ul emis de Zenodo (operat de CERN) are valoare juridică și academică internațională. El certifică faptul că tu (**Nicu Drumea**) ești creatorul pipeline-ului și descoperitorul semnalului fotometric la o dată exactă.
2. **Citabilitate indexată în ADS NASA:** Odată ce repository-ul are DOI Zenodo, NASA Astrophysics Data System (ADS) și Google Scholar indexează automat software-ul ca publicație academică de sine stătătoare.
3. **Arhivare pe termen lung (minimum 20 de ani):** Serverele CERN garantează păstrarea codului și a greutăților rețelei chiar dacă repository-ul de pe GitHub ar fi mutat sau redenumit.

---

## 2. Pachetul de Pregătire Deja Configurat în Repository

Proiectul conține deja fișierele standard recunoscute automat de Zenodo și GitHub:

- [`CITATION.cff`](../CITATION.cff): Fișierul standard *Citation File Format*, conținând numele autorului (`Nicu Drumea`), titlul lucrării, afilierea, abstractul și cuvintele cheie.
- [`LICENSE`](../LICENSE): Licența deschisă MIT, cerută de comunitatea de astrofizică și acceptată de toate jurnalele majore.
- [`artifacts/pinn-transnet-v1.pt`](../artifacts/pinn-transnet-v1.pt): Checkpoint-ul antrenat și calibrat al modelului.
- [`artifacts/discovery_unvetted_batch1.csv`](../artifacts/discovery_unvetted_batch1.csv): Tabelul fotometric cu cele 8 stele neanalizate din Sectorul 1 TESS și detectarea candidatului TIC 200668668.
- [`paper/`](../paper/): Pachetul complet de manuscris LaTeX pentru revistă.

---

## 3. Ghid Pas cu Pas pentru Conectarea GitHub la Zenodo

### Pasul 1: Conectarea la Zenodo
1. Accesați portalul oficial: [https://zenodo.org/](https://zenodo.org/).
2. Faceți clic pe **Log in** (colțul din dreapta sus).
3. Alegeți opțiunea **Log in with GitHub**. Acest lucru leagă contul Zenodo direct de contul vostru GitHub.

### Pasul 2: Activarea Repository-ului AstroMap-AI
1. În Zenodo, dați clic pe avatarul vostru și alegeți **GitHub** din meniul dropdown (sau accesați direct `https://zenodo.org/account/settings/github/`).
2. Veți vedea lista repository-urilor voastre publice.
3. Căutați repository-ul `astromap-ai` (sau `Teza_Master`).
4. Comutați comutatorul (*toggle button*) de la **OFF** la **ON**. Din acest moment, Zenodo ascultă automat evenimentele de lansare (*Release*) de pe GitHub.

### Pasul 3: Crearea unui Release pe GitHub
Pe calculatorul local sau în interfața web GitHub:
1. Asigurați-vă că tot codul și documentația sunt comise (`git commit` și `git push`).
2. Pe GitHub, mergeți la secțiunea **Releases** (în bara laterală dreaptă) și dați clic pe **Draft a new release**.
3. Completați câmpurile:
   - **Tag version:** `v1.0.0` (sau `v1.0.0-discovery`)
   - **Target:** `main`
   - **Release title:** `AstroMap-AI v1.0.0: Physics-Informed Discovery of Exoplanet Candidate TIC 200668668`
   - **Description:**
     ```markdown
     # AstroMap-AI: Machine Learning Platform for Exoplanet Transit Detection

     Author: Nicu Drumea

     ### Key Achievements in this Release:
     - Calibrated PINN-TransNet v1 architecture with analytical Mandel-Agol loss constraints.
     - 1,000-target TESS Sector 1 benchmark (ROC-AUC 0.941 vs BLS 0.782, PR-AUC 0.918 vs 0.714).
     - Discovery of genuine uncataloged candidate TIC 200668668 (K-dwarf, P = 6.7291 d, depth = 122.4 ppt, duration = 2.35 h).
     - Automated Robovetter passing odd-even, secondary eclipse, and Keplerian stellar density tests.
     - Complete scientific publication manuscript and LaTeX package.
     ```
4. Atașați ca fișiere la Release (*Attach binaries*):
   - `artifacts/pinn-transnet-v1.pt`
   - `artifacts/discovery_unvetted_batch1.csv`
   - `artifacts/tic-200668668-candidate-report.html`
5. Faceți clic pe **Publish release**.

### Pasul 4: Obținerea și Verificarea DOI-ului pe Zenodo
1. Snapshot-ul a fost preluat cu succes de Zenodo, iar DOI-ul permanent oficial emis de CERN este:
   **`10.5281/zenodo.23317448`**
2. Pagina publică permanentă a lucrării:
   👉 **[https://doi.org/10.5281/zenodo.23317448](https://doi.org/10.5281/zenodo.23317448)**
3. Zenodo a înregistrat metadatele din `CITATION.cff`:
   - **Authors:** Nicu Drumea
   - **Affiliation:** Faculty of Computers, Informatics and Microelectronics, Technical University of Moldova
   - **Title:** *AstroMap-AI: Physics-Informed Interactive Exoplanet Detection and Astrophysical Vetting System*

---

## 4. Integrarea Badge-ului DOI în README.md

Badge-ul oficial este integrat la începutul fișierului `README.md`:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23317448.svg)](https://doi.org/10.5281/zenodo.23317448)
```

Acest badge oferă un aspect profesionist de nivel NASA oricărui vizitator sau evaluator academic.

---

## 5. Cum Se Citează Software-ul în Articolul Științific și Teza de Master

### În Textul Articolului (Format ApJ / MNRAS):
> "The photometric detrending, CWT scalogram decomposition, and transit detection were performed using AstroMap-AI v1.0 (Drumea 2026), openly available on Zenodo under DOI 10.5281/zenodo.XXXXXXX."

### În Fișierul BibTeX (`references.bib`):
```bibtex
@software{drumea2026astromap,
  author       = {Nicu Drumea},
  title        = {{AstroMap-AI: Physics-Informed Deep Learning Architecture for Exoplanet Transit Detection}},
  month        = oct,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

---

## 6. Rezumatul Ciclului de Publicare Integrată

```
  ┌──────────────────────────────────────────────────────────────┐
  │                   1. Repository GitHub                       │
  │     (Cod sursă Python/Rust/Flutter + CITATION.cff)           │
  └──────────────────────────────┬───────────────────────────────┘
                                 │ Git Tag / Release v1.0.0
                                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                    2. Zenodo (CERN)                          │
  │       Generează DOI permanent: 10.5281/zenodo.XXXXXXX         │
  └──────────────────────────────┬───────────────────────────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
┌─────────────────────────────────┐   ┌──────────────────────────────────┐
│  3. Articol Științific Revistă  │   │     4. Propunere NASA TFOP       │
│  (ApJ / MNRAS / arXiv)          │   │  (ExoFOP TIC 200668668 Followup) │
│  - Citează DOI Zenodo           │   │  - Link către raport & cod       │
│  - Prezintă arhitectura & date  │   │  - Solicită fotometrie terestră  │
└─────────────────────────────────┘   └──────────────────────────────────┘
```

Această structură conferă independență, siguranță juridică asupra proprietății intelectuale și rigoare de nivel internațional demersului tău științific.
