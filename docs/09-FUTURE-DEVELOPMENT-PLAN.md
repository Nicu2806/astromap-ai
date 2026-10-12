# Plan de Dezvoltare în Continuare (Future Roadmap & Scalare)

> **Document:** Plan de dezvoltare strategică și tehnică pentru evoluția AstroMap-AI  
> **Orizont temporal:** 2026 – 2027  
> **Stadiu inițial:** MVP funcțional end-to-end complet (Inspecție fotometrică MAST, API Rust, Client Flutter cu Phase-Folded plot, Dataset versionat anti-leakage, Rețea antrenată PINN-TransNet-v1, Studiu de ablație RQ1/RQ2).

---

## 1. Context și Punctul de Plecare

AstroMap-AI a atins cu succes toate obiectivele de bază ale unui sistem funcțional de cercetare:
1. **Traseul live:** Ingestie directă a produselor SPOC reale din arhiva NASA MAST, deduplicare pe sector și detrending fotometric Savitzky-Golay.
2. **Arhitectura neurală:** Rețea hibridă `PinnTransNet` (1D flux + 2D scalogramă CWT Ricker) optimizată cu penalizare fizică Kepleriană (`physics_loss`).
3. **Validare fără compromisuri:** Dataset versionat (`data/manifest.json`), split stratificat grupat pe `tic_id` cu zero data leakage, studiu de ablație complet (`artifacts/ablation_study.json`) și model card formal (`artifacts/model_card.json`).
4. **Platforma utilizator:** Gateway public Rust cu timeout de 120s și suite de teste mock, plus client Flutter Web/Desktop dotat cu vizualizare fotometrică și pliere pe faza orbitală (**Phase-Folded Transit Curve**).

Pentru a scala acest proiect dincolo de cerințele unei teze de master standard către o **platformă de nivel academic publicabilă internațional** (ex. conferințe IEEE/AAS sau jurnalul *Monthly Notices of the Royal Astronomical Society - MNRAS*), este necesar următorul plan de dezvoltare structurat pe 4 piloni tehnologici.

---

## 2. Pilonii Strategici de Dezvoltare

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       AstroMap-AI: Pilonii de Scalare                       │
├──────────────────────┬──────────────────────┬───────────────────────────────┤
│ Pilonul 1: Big Data  │ Pilonul 2: Arhit.    │ Pilonul 3: Modele Avansate    │
│ & Ingestie Cloud     │ Asincronă & DB       │ & Fizică Diferențiabilă       │
├──────────────────────┼──────────────────────┼───────────────────────────────┤
│ - TESS Sectoare 1-70 │ - Redis Job Queue    │ - Mandel-Agol diferențiabil   │
│ - Date Kepler Q1-Q17 │ - PostgreSQL+Timescale│ - Detecție TTV (Attention)    │
│ - AWS S3 Bulk Mirror │ - Server-Sent Events │ - Gaussian Process detrending │
│ - Cache Parquet/Zarr │ - Autentificare JWT  │ - Cuantificarea incertitudinii│
└──────────────────────┴──────────────────────┴───────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Pilonul 4: Workbench Avansat (Aladin Lite HiPS, Centroid Vetting, PDF Export)│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Pilonul 1: Scalarea Masivă a Datelor & Ingestie Cloud (Big Data)

### Obiectiv
Trecerea de la catalogul de eșantionare actual (32 ținte / 50 observații) la procesarea la scară a peste 50.000 de curbe de lumină TESS și Kepler.

### Direcții de Implementare
1. **Ingestie Bulk via AWS Open Data Registry:**
   - Datele TESS SPOC sunt găzduite gratuit în bucket-ul Amazon S3 public: `s3://stpubdata/tess/public/tid/`.
   - Implementarea unui modul `astromap.bulk_ingest` care folosește `aiohttp` / `boto3` pentru a descărca paralel fișiere FITS direct din S3 fără a supraîncărca serverele MAST.
2. **Format de Stocare Optimizat (Zarr / Parquet Columnar):**
   - Fișierele FITS sunt mari și conțin header-e redundante.
   - Conversia seriilor fotometrice și a scalogramelor 2D precalculate într-un format compresibil pe coloane:
     * Serii temporale: **Parquet** (zstd compresie).
     * Scalograme CWT: **Zarr** cu chunk-uri dimensionate pentru citire GPU directă.
3. **Cross-Mission Transfer Learning (Kepler -> TESS):**
   - Antrenarea modelului pe setul masiv Kepler (misiunea primară, cadență de 30 minute, 4 ani continuu de date).
   - Fine-tuning pe datele TESS (cadență de 2 minute și 20 secunde, zgomot instrumental diferit).
   - Testarea robusteții transferului de domeniu (Domain Adaptation).

---

## 4. Pilonul 2: Arhitectură Asincronă & Persistență Enterprise

### Obiectiv
Eliminarea blocajului apelurilor HTTP sincrone (timeout 120s) și introducerea unui sistem de joburi distribuit cu persistență pe termen lung.

### Direcții de Implementare
1. **Coadă Asincronă de Procesare (Job Queue):**
   - Înlocuirea endpoint-ului sincron `POST /api/v1/analyses/:tic` cu un flux asincron bazat pe token:
     * `POST /api/v1/analyses/:tic` returnează instantaneu `HTTP 202 Accepted` cu `{"job_id": "...", "status": "queued"}`.
     * Un worker de fundal (Celery cu Redis sau Tokio Task Pool în Rust) preia sarcina de analiză/descărcare.
     * `GET /api/v1/jobs/:job_id` permite interogarea stării sau streaming în timp real via **Server-Sent Events (SSE)**.
2. **Bază de Date PostgreSQL + TimescaleDB + PostGIS:**
   - **PostGIS:** Indexare spațială sferică (coordonate ICRS RA/Dec) pentru interogări conice ultrarapide (`ST_DWithin` în spațiu ceresc).
   - **TimescaleDB:** Stocarea hiper-eficientă a vectorilor de flux și timp normalizat.
   - **PostgreSQL:** Persistența candidaților analizați, istoricul utilizatorilor și auditul metadatelor.
3. **Cache Stratificat:**
   - L1: Memorie locală LRU (în Rust / Python).
   - L2: Redis Cache distribuit (TTL 7 zile pentru produse FITS identice).

---

## 5. Pilonul 3: Modele Avansate & Fizică Diferențiabilă (Cercetare Doctorală)

### Obiectiv
Depășirea constrângerilor de bază prin integrarea legilor astrofizice directe în backpropagation și detecția fenomenelor dinamice complexe.

### Direcții de Implementare
1. **Modelul Analitic Mandel-Agol Complet Diferențiabil:**
   - Înlocuirea aproximării duratei maxime cu ecuațiile analitice complete de tranzit Mandel & Agol (2002) implementate în PyTorch/JAX diferențiabil.
   - Modelul va genera direct profilul de flux teoretic în funcție de raportul razelor $R_p/R_*$, semi-axa mare $a/R_*$, înclinație $i$ și coeficienții de limb darkening patratic $(u_1, u_2)$, calculând eroarea de reconstrucție $\mathcal{L}_{\text{profile}} = \|f_{\text{obs}} - f_{\text{model}}\|_2^2$.
2. **Detecția Variațiilor de Timp ale Tranzitului (TTV - Transit Timing Variations):**
   - Tranzitele planetare perturbate de alte planete din sistem nu apar la intervale riguros periodice.
   - Adăugarea unui modul temporal cu **Self-Attention** (Transformer Encoder) sau rețele convoluționale temporale (**TCN**) capabil să identifice tranzite aperiodice sau cvasi-periodice.
3. **Detrending cu Procese Gaussiene (Gaussian Processes):**
   - Înlocuirea filtrului Savitzky-Golay cu nuclee celerite/SHOM (Simple Harmonic Oscillator) pentru modelarea zgomotului roșu (granulație stelară, pete solare).
4. **Cuantificarea Incertitudinii (Uncertainty Estimation):**
   - Utilizarea tehnicilor Monte Carlo Dropout sau Evidential Deep Learning pentru a genera intervale de încredere (1$\sigma$, 3$\sigma$) pentru perioada și adâncimea prezise.

---

## 6. Pilonul 4: Workbench Vizual Avansat & Interoperabilitate Astronomică

### Obiectiv
Transformarea aplicației Flutter într-un instrument de clasă astronomică utilizabil zilnic de cercetători.

### Direcții de Implementare
1. **Integrare Aladin Lite v3 (HiPS Survey):**
   - Încorporarea hărții interactive Aladin Lite (DSS2, 2MASS, Gaia eDR3) prin container `HtmlElementView` în Flutter Web.
   - Afișarea conturului sectorului TESS și a stelelor vecine care pot produce contaminare luminoasă.
2. **Diagnostic de Vetting de Centroid (Centroid Motion Analysis):**
   - Una dintre cele mai frecvente cauze de falși pozitivi este o binară cu eclipse aflată pe fundal în aceeași apertură fotometrică.
   - Afișarea mișcării centrului de masă luminos al pixelilor în timpul tranzitului față de timpul din afara tranzitului.
3. **Vizualizarea Directă a Scalogramei CWT:**
   - Adăugarea unui widget tip spectrograf / heatmap 2D în Flutter care să permită utilizatorului explorarea interactivă a celor 32 de scări wavelet.
4. **Export Automatizat de Rapoarte PDF:**
   - Generarea de rapoarte de analiză în format PDF de calitate editorială (stil MNRAS/ApJ) via LaTeX sau headless Chromium.

---

## 7. Planul Operațional pe Faze (Gantt Conceptual)

| Faza | Livrabile Cheie | Durată Estimată |
|---|---|:---:|
| **Faza 5: Scalare Date & AWS Ingest** | Ingestie bulk S3, manifest extins 10k ținte, format Parquet/Zarr | Luna 1 – 2 |
| **Faza 6: Backend Asincron & DB** | Coadă Redis, PostgreSQL+PostGIS, Server-Sent Events în Rust | Luna 3 – 4 |
| **Faza 7: Fizică Mandel-Agol & TTV** | Transit profile diferențiabil, model Transformer/TCN, ablație TTV | Luna 5 – 6 |
| **Faza 8: Aladin Lite & Centroid Vetting** | Hartă HiPS în Flutter, diagnostic centroid, export rapoarte PDF | Luna 7 – 8 |

---

## 8. Criterii de Succes pentru Publicare / Finalizare Teză

1. **Performanță pe catalog mare (>5.000 curbe):**
   - PR-AUC $\ge 0.88$ pe set de test independent ne-văzut.
   - FPR $\le 0.05$ la Recall de 95% (reducerea cu cel puțin 40% a falșilor pozitivi față de BLS standard).
2. **Reproducibilitate 100%:**
   - Un singur script CLI reproduce întregul tabel de ablație și generează graficele din teză.
3. **Timp de răspuns:**
   - Inferența pe o curbă de lumină procesată: $< 150\text{ ms}$ pe CPU, $< 25\text{ ms}$ pe GPU.
