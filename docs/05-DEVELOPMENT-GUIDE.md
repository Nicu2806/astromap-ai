# Ghid de dezvoltare

## 1. Cerințe locale

Versiunile țintă:

- Windows 10/11 sau Linux;
- Python 3.11+;
- Rust stable și Cargo;
- Flutter stable cu Dart 3;
- Git;
- acces internet pentru NASA MAST;
- Docker opțional pentru deployment;
- GPU NVIDIA opțional pentru antrenare, nu pentru BLS sau UI.

Verificare rapidă PowerShell:

```powershell
python --version
rustc --version
cargo --version
flutter --version
git --version
docker --version
```

Absența Docker nu blochează dezvoltarea locală a celor trei servicii.

## 2. Inițializarea Python

Din rădăcina repository-ului:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".\python[api,dev,data]"
```

Extra-uri:

- `api`: FastAPI și Uvicorn;
- `dev`: pytest, HTTP test client și Ruff;
- `data`: Pandas și PyArrow pentru viitorul pipeline de dataset;
- Lightkurve, Astropy, NumPy, SciPy, PyTorch și Pydantic sunt dependențe de bază.

Nu instala pachetul doar în Python-ul global dacă VS Code folosește `.venv`. Interpreterul selectat trebuie să fie `.venv/Scripts/python.exe`.

## 3. Inițializarea Rust

Cargo rezolvă dependențele din [`apps/api-rust/Cargo.toml`](../apps/api-rust/Cargo.toml):

```powershell
cargo build --manifest-path apps/api-rust/Cargo.toml
```

Nu este necesară o bază de date în MVP.

## 4. Inițializarea Flutter

```powershell
Push-Location apps/client-flutter
flutter pub get
flutter doctor
Pop-Location
```

Proiectul are target-uri Web și Windows. Web este calea principală verificată.

## 5. Pornirea serviciilor

### Automat

```powershell
.\scripts\start-dev.ps1
```

Scriptul pornește procese separate și nu le gestionează ciclul de viață după lansare. Pentru debugging este preferabilă pornirea manuală.

### Manual

Terminal Python:

```powershell
.\.venv\Scripts\python.exe -m uvicorn astromap.api:app `
  --host 127.0.0.1 --port 8001
```

Terminal Rust:

```powershell
$env:RUST_LOG='api_rust=info,tower_http=info'
cargo run --manifest-path apps/api-rust/Cargo.toml
```

Terminal Flutter:

```powershell
Push-Location apps/client-flutter
flutter run -d web-server --web-hostname 127.0.0.1 --web-port 8088
```

URL-uri:

- UI: `http://127.0.0.1:8088`;
- API public: `http://127.0.0.1:8080`;
- Swagger intern Python: `http://127.0.0.1:8001/docs`.

## 6. Variabile de mediu pentru dezvoltare

```powershell
# Python: activează un checkpoint existent
$env:ASTROMAP_CHECKPOINT='C:\path\model.pt'

# Rust: schimbă serviciul intern
$env:ASTROMAP_INFERENCE_URL='http://127.0.0.1:8001'
$env:ASTROMAP_API_PORT='8080'
$env:RUST_LOG='api_rust=debug,tower_http=info'

# Flutter: valoare compile-time
flutter run -d web-server `
  --dart-define=ASTROMAP_API_URL=http://127.0.0.1:8080
```

După schimbarea `ASTROMAP_CHECKPOINT`, repornește Python. După schimbarea `ASTROMAP_API_URL`, rebuild/restart Flutter.

## 7. Fluxuri comune de modificare

### 7.1 Schimbare de preprocesare

1. modifică [`pipeline.py`](../python/src/astromap/pipeline.py);
2. adaugă teste deterministe în `python/tests/test_pipeline.py`;
3. verifică formele `512` și `32x512` ori actualizează modelul;
4. rulează toate testele Python;
5. documentează efectul științific și necesitatea unei noi ablații.

O schimbare de preprocesare invalidează comparabilitatea cu checkpoint-uri antrenate pe versiunea veche.

### 7.2 Schimbare de model

1. modifică `PinnTransNet` și testul formelor;
2. păstrează ori versionează contractul celor trei parametri;
3. rulează smoke training și reload checkpoint;
4. nu promova checkpoint-ul fără evaluare reală;
5. actualizează metodologia și model card-ul viitor.

### 7.3 Adăugare endpoint

1. decide dacă logica aparține Python sau Rust;
2. adaugă endpoint intern Python pentru operații științifice;
3. adaugă proxy/validare Rust;
4. actualizează `schemas/openapi.yaml`;
5. actualizează modelele/clientul Flutter;
6. adaugă teste de contract;
7. verifică end-to-end cu date reale.

### 7.4 Adăugare câmp în `Prediction`

Ordinea obligatorie:

1. `python/src/astromap/inference.py`;
2. testele API Python;
3. `apps/client-flutter/lib/models.dart`;
4. UI și testele Dart;
5. OpenAPI și documentația API.

Rust proxiază predicția ca `serde_json::Value`, deci compilarea Rust nu detectează câmpuri lipsă din DTO-ul Flutter.

## 8. Convenții de cod

### Python

- tipuri explicite pentru API-urile publice;
- dataclass frozen pentru structuri transferate;
- docstrings pentru algoritmi și limite;
- fără I/O de rețea la import;
- dependency injection pentru testele MAST;
- fără fallback sintetic în endpoint-ul live.

### Rust

- handler-ele returnează `Result<Json<_>, ApiError>`;
- validează inputul la granița publică;
- păstrează timeout explicit pentru downstream;
- nu utiliza `unwrap()` în handler-ele de producție;
- JSON științific rămâne proprietatea Python.

### Flutter

- modele tipizate și factory `fromJson`;
- `const` unde este posibil;
- tratează loading, empty, error și success;
- comenzile lungi dezactivează butonul pentru a evita duplicate;
- nu transforma un scor în altă unitate sau semantică.

## 9. Date și artefacte

Nu adăuga în Git:

- FITS;
- checkpoint-uri `.pt`;
- scalograme `.npz`;
- Parquet de dimensiuni mari;
- cache Lightkurve;
- rapoarte generate pentru rulări locale.

Pentru un experiment real, salvează separat:

- manifestul și checksum-ul datelor;
- commit SHA;
- config complet;
- seed-uri;
- versiuni de biblioteci/CUDA;
- checkpoint și checksum;
- metrici și pragul ales pe validation.

## 10. Troubleshooting local

### `No SPOC light curve found`

- verifică TIC ID;
- verifică dacă sectorul aparține țintei prin endpoint-ul observations;
- încearcă fără sector pentru primul produs disponibil;
- verifică accesul la MAST.

### Timeout Rust

- prima descărcare poate depăși conexiuni lente;
- verifică logul Python;
- verifică cache-ul Lightkurve;
- nu crește timeout-ul fără limită; pentru operații lungi implementează job queue.

### Flutter arată `DEMO CATALOG`

- Rust nu este accesibil la URL-ul compile-time;
- verifică `ASTROMAP_API_URL` și CORS;
- fallback-ul conține doar metadata de afișare, nu permite inventar/inferență fără backend.

### Checkpoint configurat, dar serviciul eșuează

- calea trebuie să existe în filesystem-ul procesului/containerului;
- checkpoint-ul trebuie să corespundă exact arhitecturii curente;
- nu folosi checkpoint-ul smoke ca model validat.

## 11. Curățarea proceselor locale

Identifică porturile înainte de oprire:

```powershell
Get-NetTCPConnection -LocalPort 8001,8080,8088 -State Listen |
  Select-Object LocalPort,OwningProcess
```

Oprește numai procesele proiectului, fără comenzi globale care pot închide alte sesiuni Python, Cargo sau Flutter.