# Ghid de Instrucțiuni și Context: AstroMap-AI
> **Destinat:** Următorului Agent AI sau Dezvoltator software/cercetător care preia sau contribuie la proiect.  
> **Data actualizării:** Octombrie 2026  
> **Status proiect:** MVP funcțional end-to-end (Screening fotometric live cu BLS pe date NASA MAST + Arhitectură neurală PINN pregătită pentru antrenament real).

---

## 1. Contextul Proiectului & Viziunea Tezei de Master

### Ce este AstroMap-AI?
AstroMap-AI este o platformă completă de cercetare și screening pentru identificarea automată a semnalelor de tranzit asociate exoplanetelor din curbele de lumină obținute de telescopul spațial **NASA TESS** (Transiting Exoplanet Survey Satellite) prin misiunea SPOC (Science Processing Operations Center).

### Scopul Tezei
1. **Contribuție științifică:** Dezvoltarea și evaluarea unei metode hibride de învățare profundă ghidată de fizică (**PINN-TransNet** – *Physics-Informed Neural Network for Transit Detection*), comparată riguros cu metoda astrofizică clasică standard (**BLS** – *Box Least Squares*).
2. **Platformă completă de cercetare:** Furnizarea unui sistem software modular compus din:
   - Motor de procesare fotometrică și inferență AI în **Python**;
   - API Gateway public de înaltă performanță în **Rust**;
   - Workbench interactiv de vizualizare astronomică în **Flutter** (Web și Desktop).

> [!IMPORTANT]
> **AstroMap-AI este un sistem de screening și cercetare, nu o metodă de confirmare independentă a unei planete.**  
> O predicție pozitivă semnalează prezența unui candidat cu trăsături compatibile cu un tranzit fotometric și necesită analize suplimentare (ephemeris match, centroid vetting, spectroscopie radială).

---

## 2. Ideea Științifică și Algoritmică

Proiectul abordează limitările algoritmilor clasici (care sunt lenți pe serii lungi de date și sensibili la zgomot stelar/gap-uri) și ale modelelor pur statistice de tip "black-box" (care pot clasifica eronat stele binare cu eclipse sau zgomot instrumental ca fiind planete):

```
                   Curba TESS (Flux fotometric vs Timp BTJD)
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        Ramura Temporală 1D                    Ramura Timp-Frecvență 2D
       (Flux resampled la 512)            (Scalogramă CWT Ricker: 32x512)
                   │                                     │
                   ▼                                     ▼
        1D CNN Feature Extractor              2D CNN Feature Extractor
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      │ (Concatenare caracteristici)
                                      ▼
                             Dense / Linear Layers
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
     Scor Candidat Tranzit (0..1)             Parametri Fizici (Softplus)
       [Clasificare binară]                   - Perioadă orbitală estimată (P)
                                              - Durată tranzit (T_dur)
                                              - Adâncime relativă tranzit (depth)
                                                         │
                                                         ▼
                                                Physics-Loss Penalty
                                          (Durata maximă pe orbită Kepleriană)
```

1. **Reprezentare Dual-Domain (1D + 2D):**
   - **1D Flux Temporal:** Surprinde morfologia clasică în formă de „U” a tranzitului și raportul semnal/zgomot local.
   - **2D Scalogramă CWT (Continuous Wavelet Transform):** Calculată prin wavelet-ul Ricker (Mexican Hat) pe 32 de scări logaritmice. Aceasta localizează tranzitul simultan în timp și frecvență, permițând discriminarea zgomotului stelar de joasă frecvență de scăderile abrupte caracteristice tranzitelor.
2. **Constrângere Fizică (Physics-Informed Loss):**
   - Pentru o orbită circulară centrală în jurul unei stele de masă și rază solară ($M_*, R_*$), durata maximă teoretică a tranzitului este dictată de legile lui Kepler:
     $$T_{\text{dur, max}} \approx \frac{R_*}{\pi a} P = \left(\frac{3}{\pi^2 G \rho_*}\right)^{1/3} P^{1/3}$$
   - Modelele care prevăd durate fizic incompatibile cu perioada estimată primesc o penalizare explicită în funcția de loss (`physics_loss`), împiedicând modelul să învețe artefacte instrumentale nerealiste.
3. **BLS ca Baseline și Fallback Permanent:**
   - Algoritmul Box Least Squares (Astropy) este baseline-ul astrofizic de referință. Dacă nu există un model AI antrenat sau checkpoint încărcat, sistemul rulează BLS pe curba curățată și returnează parametrii și un `bls_screening_score`.

---

## 3. Starea Reală a Implementării (Ground Truth)

Pentru a nu face presupuneri eronate, oricine lucrează pe acest repository trebuie să cunoască starea exactă a codului:

| Componentă / Funcționalitate | Stare | Detalii Tehnice |
|---|---|---|
| **Interogare NASA MAST** | ✅ **Funcțională (Live)** | `mast.py` descarcă produse SPOC reale pentru orice TIC valid prin biblioteca Lightkurve. Are cache în memorie și pe disc. |
| **Pipeline fotometric** | ✅ **Funcțional** | Eliminare NaN, filtrare `quality == 0`, detrending Savitzky-Golay, normalizare mediană la 1.0, mascare gap-uri, resampling la 512 puncte, generare scalogramă `32x512`. |
| **Algoritm BLS** | ✅ **Funcțional (Live)** | Rulează `BoxLeastSquares` din Astropy, determină perioada, durata, adâncimea și calculează raportul SNR ca scor de screening. |
| **Arhitectură PyTorch (`PinnTransNet`)** | ✅ **Implementată** | Modelele 1D/2D, concatenarea, softplus pentru parametri pozitivi și `physics_loss` sunt implementate și testate unitar. |
| **Antrenare pe date reale** | ⚠️ **NEIMPLEMENTAT** | `training.py` conține doar un script demonstrativ (*smoke test*) pe curbe sintetice generate matematic, folosit exclusiv pentru a valida bucla de antrenament și formatul fișierului `.pt`. |
| **Checkpoint AI validat** | ❌ **LIPSEȘTE** | **Nu există încă un checkpoint antrenat pe catalogul real de TOI/TCE.** Sistemul rulează implicit pe BLS (`score_kind = "bls_screening_score"`). |
| **API Gateway Rust** | ✅ **Funcțional** | Construit cu Axum și Tokio, expune `/api/v1/targets`, `/api/v1/targets/:tic/observations` și `/api/v1/analyses/:tic`. Include timeout de 120s pentru descărcări MAST lente. |
| **Client Flutter** | ✅ **Funcțional** | Interfață Web și Desktop cu selector de ținte, interogare dinamică a sectoarelor disponibile, grafic al fluxului normalizat și afișarea completă a provenienței. |
| **Containere Docker** | 🔶 **Configurate** | `docker-compose.yml` și fișierele Dockerfile sunt create în `infra/`, dar necesită validare pe o mașină Linux cu Docker activ. |

> [!CAUTION]
> **Interdicție semantică:** Nu denumiți NICIODATĂ un rezultat generat de BLS ca fiind „probabilitate AI” și nu pretindeți că sistemul a „confirmat o nouă exoplanetă”.  
> Câmpul `score_kind` distinge explicit între `bls_screening_score` și `unvalidated_ai_probability`.

---

## 4. Arhitectura Celor Trei Componente

Separarea pe tehnologii este strictă și bine definită:

```text
Teza_Master/
├── python/                  # BACKEND ȘTIINȚIFIC (FastAPI :8001)
│   └── src/astromap/        # MAST ingest, Astropy BLS, PyTorch PINN, Preprocessing
│   └── tests/               # 9 teste unitare și de integrare
├── apps/
│   ├── api-rust/            # PUBLIC API GATEWAY (Axum :8080)
│   │   ├── src/             # Rute HTTP, validare, catalog ținte, proxy către Python
│   │   └── tests/           # Teste de integrare Rust
│   └── client-flutter/      # CLIENT WORKBENCH (Flutter :8088)
│       ├── lib/             # Interfață grafică, grafice fotometrice, modele JSON
│       └── test/            # Teste widget și de contract API
├── schemas/
│   └── openapi.yaml         # Contractul OpenAPI 3.0 al API-ului public
├── configs/
│   └── training.yaml        # Configurație de experiment și hiperparametri
├── docs/                    # Documentația completă pe 8 module tematice
├── scripts/
│   └── start-dev.ps1        # Script PowerShell pentru lansare automată locală
└── .github/
    └── copilot-instructions.md # Reguli de comportament pentru agenți AI
```

### Regula de Aur a Separării Responsabilităților:
1. **Python deține adevărul științific:** Orice operațiune cu fișiere FITS, fotometrie, transformate wavelet, algoritmi BLS sau inferență PyTorch aparține exclusiv pachetului `astromap`.
2. **Rust este poarta de intrare publică:** Rust nu parsează FITS și nu calculează statistici astrofizice. Rolul său este securitatea, rutarea, validarea parametrilor de intrare, rate limiting-ul și decuplarea clientului de backend-ul intern.
3. **Flutter este prezentarea:** Flutter nu modifică datele primite, nu filtrează punctele și nu calculează probabilități. El afișează fidel vectorul de date, metadatele și proveniența primite prin API.

---

## 5. Ghid de Lucru Pas cu Pas pentru Următorul Agent sau Developer

Când primiți o sarcină pe acest repository, respectați cu strictețe următorul protocol de lucru:

### Pasul 1: Inspectarea Contextului și a Stării Git
- Rulați `git status` pentru a vedea dacă există modificări nesalvate. Nu suprascrieți fișiere fără a înțelege ce conțin.
- Consultați documentul specific sarcinii din directorul [`docs/`](docs/README.md) (ex: `03-SCIENTIFIC-METHODOLOGY.md` pentru știință, `04-API-CONTRACTS.md` pentru endpoint-uri).

### Pasul 2: Verificarea Stării de Sănătate a Mediului (Smoke Testing)
Înainte de a efectua orice modificare, asigurați-vă că testele existente trec:
```powershell
# Verificare Python
.\.venv\Scripts\python.exe -m pytest python/tests
.\.venv\Scripts\python.exe -m ruff check python/src python/tests

# Verificare Rust
cargo test --manifest-path apps/api-rust/Cargo.toml
cargo clippy --manifest-path apps/api-rust/Cargo.toml --all-targets -- -D warnings

# Verificare Flutter
Push-Location apps/client-flutter
flutter test
flutter analyze
Pop-Location
```

### Pasul 3: Reguli de Dezvoltare (TDD & Modificări Minime)
- Scrieți întâi un test unitar care eșuează înainte de a implementa o funcționalitate nouă sau de a repara un bug.
- Modificați exclusiv componenta responsabilă (nu mutați logică între servicii).
- **Dacă modificați structura unui răspuns JSON (`Prediction`):**
  1. Actualizați dataclass-ul în `python/src/astromap/inference.py`.
  2. Actualizați testul unitar în `python/tests/test_inference.py`.
  3. Actualizați modelul Dart în `apps/client-flutter/lib/models.dart`.
  4. Actualizați schema `schemas/openapi.yaml`.
  5. Actualizați documentul `docs/04-API-CONTRACTS.md`.

---

## 6. Reguli Stricte și Anti-Patterns de Evitat

> [!WARNING]
> Orice abatere de la regulile de mai jos compromite validitatea științifică a tezei de master.

1. **ANTI-PATTERN: Data Leakage la împărțirea datasetului**
   - **INTERZIS:** Să împărțiți datele în train/validation/test la nivel de ferestre fotometrice sau cadențe.
   - **OBLIGATORIU:** Împărțirea se face strict la nivel de identificator stelar (**TIC ID** sau **KIC ID**). Toate sectoarele și toate observațiile aceleiași stele trebuie să se afle în aceeași partiție!
2. **ANTI-PATTERN: Fallback sintetic la erorile reale**
   - **INTERZIS:** Dacă NASA MAST nu găsește curba de lumină sau conexiunea cade, să returnați o curbă simulată sau un mock.
   - **OBLIGATORIU:** Returnați codul de eroare HTTP 404 sau 502 cu un mesaj clar („No SPOC light curve found for target”). Datele simulate sunt permise exclusiv în teste offline.
3. **ANTI-PATTERN: Ignorarea provenienței datelor**
   - Fiecare rezultat returnat trebuie să conțină: `target_id`, `mission`, `sector`, `cadence_count`, `provenance_uri` și `score_kind`.
4. **ANTI-PATTERN: Hardcodarea rezultatelor**
   - **INTERZIS:** Să hardcodați scoruri sau parametri pentru stele celebre (ex: TOI 700). Orice valoare trebuie să fie produsul real al analizei BLS sau al rețelei neurale.
5. **ANTI-PATTERN: Comiterea de fișiere voluminoase în Git**
   - Nu adăugați fișiere FITS (`*.fits`), modele salvate (`*.pt`), baze de date Parquet mari sau fișiere temporare în Git. Directorul `artifacts/` este ignorat prin `.gitignore`.

---

## 7. Ghid Rapid de Comenzi Locale

### Lansare automată (toate serviciile)
```powershell
.\scripts\start-dev.ps1
```
Aplicația se deschide în browser la `http://127.0.0.1:8088`.

### Lansare manuală (în terminale separate)
- **Terminal 1 (Python Scientific Engine):**
  ```powershell
  .\.venv\Scripts\python.exe -m uvicorn astromap.api:app --host 127.0.0.1 --port 8001 --reload
  ```
- **Terminal 2 (Rust Public Gateway):**
  ```powershell
  $env:RUST_LOG="api_rust=debug,tower_http=info"
  cargo run --manifest-path apps/api-rust/Cargo.toml
  ```
- **Terminal 3 (Flutter Client):**
  ```powershell
  Push-Location apps/client-flutter
  flutter run -d web-server --web-port 8088
  Pop-Location
  ```

### Antrenament smoke și raport demonstrativ
```powershell
# Antrenează rețeaua pe curbe sintetice (validează arhitectura)
.\.venv\Scripts\astromap-train.exe --epochs 2 --output artifacts/pinn-transnet-demo.pt

# Generează un raport HTML de analiză pentru TOI 700
.\.venv\Scripts\astromap-report.exe --tic-id 150428135 --sector 3 --output artifacts/toi-700-report.html
```

---

## 8. Handoff & Roadmap Prioritizat (Ce urmează de implementat)

Pentru a finaliza teza de master la standarde academice ridicate, prioritățile sunt structurate clar:

### 🎯 Prioritatea P0: Crearea Datasetului Real Etichetat (Traseul Critic)
1. **Script de ingestie din NASA Exoplanet Archive:** Descărcarea tabelelor oficiale TOI (TESS Objects of Interest) și TCE (Threshold Crossing Events).
2. **Etichetare clară:** Confirmate (CP), Candidați (PC), Eclipsing Binaries (EB), Falși Pozitivi instrumentali (FP).
3. **Partitionare fără scurgeri:** Gruparea strictă pe TIC ID în Train (70%), Validation (15%), Test blocat (15%).
4. **Manifest versionat:** Salvarea metadatelor într-un fișier `manifest.parquet` cu checksum SHA-256.

### 🎯 Prioritatea P1: Antrenarea Modelului Real & Validare Benchmark
1. **PyTorch Dataset/DataLoader:** Citirea curbelor preprocesate din manifest.
2. **Antrenare cu ablații documentate:**
   - Model doar 1D (flux temporal).
   - Model hibrid 1D + 2D (flux + scalogramă CWT).
   - Model cu și fără `physics_loss`.
3. **Metrici de evaluare pe Test Set-ul blocat:**
   - ROC-AUC și PR-AUC (Precision-Recall Area Under Curve – esențială pe date dezechilibrate).
   - False Positive Rate la 95% Recall (standardul NASA Kepler/TESS).
   - Comparație directă față de baseline-ul Astropy BLS.
4. **Salvare Checkpoint aprobat:** Generarea `pinn-transnet-v1.pt` și activarea lui prin variabila de mediu `ASTROMAP_CHECKPOINT`.

### 🎯 Prioritatea P2: Arhitectură Operațională Asincronă
1. **Task Queue (Redis + Celery sau Tokio background worker):** Transformarea endpoint-ului de analiză din sincron în asincron (POST inițiază un `job_id`, GET interoghează starea prin polling sau Server-Sent Events).
2. **Persistență în PostgreSQL:** Salvarea analizelor efectuate pentru a nu reinteroga MAST la fiecare refresh de pagină.

### 🎯 Prioritatea P3: Rafinarea Interfeței Grafice (Flutter)
1. **Grafic Phase-Folded:** Afișarea curbei de lumină pliată pe perioada determinată de tranzit.
2. **Vizualizarea scalogramei CWT:** Adăugarea unei reprezentări tip heatmap a transformatei wavelet 2D în UI.
3. **Curățarea claselor demo nefolosite** rămase în `apps/client-flutter/lib/main.dart`.

---

## 9. Indexul Documentației Tehnice Detaliate

Toate detaliile aprofundate sunt organizate în directorul [`docs/`](docs/README.md):

| Document | Descriere |
|---|---|
| [**`01-PROJECT-OVERVIEW.md`**](docs/01-PROJECT-OVERVIEW.md) | Viziune, scop, terminologie astronomică și componente. |
| [**`02-ARCHITECTURE.md`**](docs/02-ARCHITECTURE.md) | Diagrame Mermaid, fluxuri de date, secvențe și ownership. |
| [**`03-SCIENTIFIC-METHODOLOGY.md`**](docs/03-SCIENTIFIC-METHODOLOGY.md) | Matematica detrending-ului, CWT Ricker, BLS, PINN și Physics Loss. |
| [**`04-API-CONTRACTS.md`**](docs/04-API-CONTRACTS.md) | Specificațiile exacte ale DTO-urilor, rutelor HTTP și codurilor de eroare. |
| [**`05-DEVELOPMENT-GUIDE.md`**](docs/05-DEVELOPMENT-GUIDE.md) | Ghid complet de instalare, depanare și bune practici de cod. |
| [**`06-DEPLOYMENT-OPERATIONS.md`**](docs/06-DEPLOYMENT-OPERATIONS.md) | Configurare Docker Compose, variabile de mediu, runbook GPU/Cloud. |
| [**`07-TESTING-QUALITY.md`**](docs/07-TESTING-QUALITY.md) | Matricea de teste, proceduri CI/CD și checklist de calitate. |
| [**`08-HANDOFF-ROADMAP.md`**](docs/08-HANDOFF-ROADMAP.md) | Handoff formal, etapele detaliate A-E și analiza datoriilor tehnice. |
| [**`09-FUTURE-DEVELOPMENT-PLAN.md`**](docs/09-FUTURE-DEVELOPMENT-PLAN.md) | Planul strategic extins: Big Data TESS/Kepler, Mandel-Agol diferențiabil, TTV și Aladin Lite. |

---
*Pentru întrebări rapide sau ghidare în sesiunea de cod, consultați oricând și fișierul de instrucțiuni pentru agenți din [`.github/copilot-instructions.md`](.github/copilot-instructions.md).*
