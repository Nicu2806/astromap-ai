# AstroMap-AI

AstroMap-AI este un MVP de cercetare end-to-end pentru detectarea tranzitelor de exoplanete. Proiectul combină procesare fotometrică și PyTorch în Python, un API public Axum în Rust și un workbench științific Flutter Web/Windows.

Documentația tehnică și ghidul de handoff sunt indexate în [docs/README.md](docs/README.md).

## Ce este implementat

- descărcare implicită de produse SPOC reale din NASA MAST, cu misiune, sector, URI și număr de cadențe;
- pipeline determinist: quality mask, detrending Savitzky-Golay, normalizare, resampling și scalogramă Ricker;
- rețea dual-domain 1D/2D, head de clasificare/regresie și constrângere fizică de durată;
- adaptor opțional Lightkurve pentru curbe SPOC din MAST;
- training smoke reproductibil și checkpoint-uri PyTorch;
- serviciu intern FastAPI și API gateway Rust;
- inventar MAST deduplicat, selector de sector, catalog, proiecție cerească și curbă fotometrică în Flutter;
- raport HTML imprimabil, OpenAPI, teste, CI și imagini Docker.

Analiza live folosește curbe reale TESS/SPOC și un baseline BLS dacă nu este configurat un checkpoint. Scorul BLS este pentru screening și nu este o probabilitate AI validată. Datele sintetice sunt folosite numai în testele offline și în smoke training. Pentru rezultatele tezei trebuie construit manifestul TESS, antrenat modelul și rulate ablațiile din [PLAN_COMPLET_ASTROMAP_AI.md](PLAN_COMPLET_ASTROMAP_AI.md).

## Cerințe

- Python 3.11+
- Rust stable
- Flutter stable
- Docker + NVIDIA Container Toolkit doar pentru deployment containerizat/GPU

## Pornire locală pe Windows

```powershell
.\scripts\start-dev.ps1
```

Apoi deschide `http://localhost:8088`. Serviciile sunt:

- inference Python: `http://127.0.0.1:8001/docs`
- API Rust: `http://127.0.0.1:8080/health/ready`
- Flutter Web: `http://127.0.0.1:8088`

Pornire manuală, în trei terminale:

```powershell
.\.venv\Scripts\python.exe -m uvicorn astromap.api:app --host 127.0.0.1 --port 8001
cargo run --manifest-path apps/api-rust/Cargo.toml
Set-Location apps/client-flutter; flutter run -d web-server --web-port 8088
```

## Verificări

```powershell
.\.venv\Scripts\python.exe -m pytest python/tests
.\.venv\Scripts\python.exe -m ruff check python/src python/tests
cargo clippy --manifest-path apps/api-rust/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path apps/api-rust/Cargo.toml
Set-Location apps/client-flutter; flutter analyze; flutter test; flutter build web
```

## Training și raport

Training-ul inclus este un smoke test pe date sintetice. El verifică bucla și formatul checkpoint-ului:

```powershell
.\.venv\Scripts\astromap-train.exe --epochs 2 --output artifacts/pinn-transnet-demo.pt
.\.venv\Scripts\astromap-report.exe --tic-id 261136679 --output artifacts/tic-261136679.html
```

Dependențele MAST sunt instalate implicit. Extra-ul `data` adaugă formatele tabulare pentru construirea datasetului:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".\python[data]"
```

Datele FITS, produsele procesate și checkpoint-urile sunt excluse din Git. Păstrează checksum-uri și manifeste versionate pentru fiecare experiment.

Lightkurve păstrează local produsele descărcate în cache. Prima analiză necesită acces la internet și poate dura; analizele ulterioare ale aceluiași produs reutilizează cache-ul.

Interfața încarcă produsele SPOC disponibile și permite alegerea explicită a sectorului. Contractele publice echivalente sunt:

```text
GET  /api/v1/targets/150428135/observations
POST /api/v1/analyses/150428135?sector=3
```

Pentru a activa un checkpoint antrenat și auditat, setează `ASTROMAP_CHECKPOINT` la calea fișierului `.pt`. În lipsa lui, serviciul folosește transparent baseline-ul BLS și etichetează scorul ca atare. Un checkpoint de smoke training nu trebuie folosit pentru concluzii astronomice.

## Docker

```powershell
docker compose -f infra/compose/docker-compose.yml up --build
```

Flutter este expus pe `8088`, API-ul pe `8080`, iar inference pe `8001`. Docker nu este instalat în mediul în care a fost generat proiectul, deci configurația este pregătită dar nu a fost executată local.

## Structură

- `python/`: pipeline, model, inference, training și rapoarte;
- `apps/api-rust/`: gateway și catalog API;
- `apps/client-flutter/`: aplicația web/desktop;
- `schemas/`: contract OpenAPI;
- `configs/`: configurații experimentale;
- `infra/`: imagini și orchestrare;
- `docs/`: planul și deciziile proiectului.

## Limitări științifice

O probabilitate produsă de sistem este un scor de screening, nu confirmarea unei planete. Modelul final trebuie evaluat fără target leakage, pe un test set blocat, cu PR-AUC, FPR la 95% recall, calibrare, intervale bootstrap și comparație BLS. Datele sintetice nu pot înlocui validarea pe TESS/Kepler.
