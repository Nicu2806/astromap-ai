# AstroMap-AI Project Guidelines

## Context obligatoriu

AstroMap-AI este un proiect de teză pentru detectarea tranzitelor de exoplanete din curbe TESS/SPOC. Citește [docs/README.md](../docs/README.md) și documentul tematic relevant înainte de modificări substanțiale.

Starea reală a MVP-ului:

- Python deține ingestia MAST, procesarea științifică, BLS, PyTorch și raportarea.
- Rust/Axum este gateway-ul public și nu reimplementează algoritmi astrofizici.
- Flutter este clientul de inspecție și nu calculează predicții.
- Datele live sunt reale; fixture-urile sintetice sunt permise numai în teste și smoke training.
- Fără `ASTROMAP_CHECKPOINT`, sistemul rulează BLS și trebuie să emită `score_kind=bls_screening_score`.
- Un checkpoint încărcat este `unvalidated_ai_probability` până la validarea formală pe test set blocat.

## Reguli științifice

- Nu numi un scor BLS „probabilitate AI”. Nu afirma că sistemul confirmă o planetă.
- Păstrează proveniența: TIC ID, misiune, sector, număr de cadențe, URI MAST și versiunea modelului/metodei.
- Nu introduce fallback sintetic în traseul live. Eșecurile MAST trebuie returnate explicit.
- Orice split de dataset trebuie făcut pe TIC/KIC ID înainte de ferestre; toate sectoarele aceleiași stele rămân în aceeași partiție.
- Nu raporta câștiguri ale CWT sau physics loss fără ablație pe același split și intervale de încredere.
- BLS rămâne baseline obligatoriu pentru orice model neural.

Vezi [metodologia științifică](../docs/03-SCIENTIFIC-METHODOLOGY.md).

## Limitele componentelor

- `python/src/astromap/`: FITS/MAST, preprocesare, CWT, model, inferență, training și rapoarte.
- `apps/api-rust/`: validare HTTP, catalog, proxy/orchestrare, CORS și erori publice.
- `apps/client-flutter/`: stare UI, selector țintă/sector, grafice și prezentarea provenienței.
- `schemas/openapi.yaml`: contractul API public; actualizează-l odată cu rutele Rust.
- `configs/training.yaml`: configurație țintă pentru experimente; nu presupune că runner-ul smoke consumă toate câmpurile.

Nu muta logică științifică în Rust sau Dart. Nu cupla antrenarea de serverul HTTP.

## Flux de lucru

1. Identifică producătorul datelor sau comportamentului greșit.
2. Adaugă ori actualizează un test focalizat.
3. Fă modificarea minimă în componenta proprietară.
4. Rulează verificarea focalizată, apoi suita componentei.
5. Pentru schimbări de contract, validează traseul Python -> Rust -> Flutter și actualizează OpenAPI/documentația.
6. Nu modifica artefacte FITS, checkpoint-uri sau date generate în Git.

## Comenzi obligatorii

Python:

```powershell
.\.venv\Scripts\python.exe -m pytest python/tests
.\.venv\Scripts\python.exe -m ruff check python/src python/tests
```

Rust:

```powershell
cargo fmt --manifest-path apps/api-rust/Cargo.toml -- --check
cargo clippy --manifest-path apps/api-rust/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path apps/api-rust/Cargo.toml
```

Flutter:

```powershell
Push-Location apps/client-flutter
dart format --output=none --set-exit-if-changed lib test
flutter analyze
flutter test
flutter build web --release
Pop-Location
```

Vezi [ghidul de dezvoltare](../docs/05-DEVELOPMENT-GUIDE.md) și [testarea](../docs/07-TESTING-QUALITY.md).

## Convenții

- Python 3.11+, type hints, frozen dataclasses pentru contracte, Ruff și pytest.
- Rust stable, Serde DTO-uri, `Result<..., ApiError>`, rustfmt și Clippy fără warnings.
- Dart 3/Flutter stable, modele tipizate, `const` unde este posibil și stări loading/empty/error/success.
- Păstrează API v1 compatibil sau documentează explicit schimbarea incompatibilă.
- Folosește UTC și unități explicite; coordonatele sunt grade ICRS, timpul fotometric trebuie să-și declare scala/formatarea.

## Limitări cunoscute

- Nu există încă un checkpoint antrenat și validat pe catalog real etichetat.
- Catalogul public Rust conține doar trei ținte TIC verificate.
- Inferența este sincronă; nu există Redis/PostgreSQL sau job queue.
- Docker Compose este pregătit, dar nu a fost validat în mediul Windows curent.
- `apps/client-flutter/lib/main.dart` păstrează clase demo nefolosite; entry point-ul real este `AstroMapApp`.

Nu remedia aceste limitări incidental într-o schimbare fără legătură. Prioritățile sunt în [handoff și roadmap](../docs/08-HANDOFF-ROADMAP.md).