# Testare și calitate

## 1. Obiective

Testarea trebuie să protejeze trei proprietăți:

1. corectitudinea numerică și determinismul pipeline-ului;
2. compatibilitatea contractului Python -> Rust -> Flutter;
3. onestitatea semantică a rezultatului BLS versus AI.

## 2. Matricea actuală

| Componentă | Teste curente | Instrumente |
|---|---:|---|
| Python | 9 | pytest, FastAPI TestClient, Ruff |
| Rust | 4 | cargo test, Clippy, rustfmt |
| Flutter | 2 | flutter_test, MockClient, flutter analyze |

Numărul este informativ și se va schimba; criteriul este acoperirea comportamentului, nu conservarea numărului.

## 3. Python

Comenzi:

```powershell
.\.venv\Scripts\python.exe -m pytest python/tests
.\.venv\Scripts\python.exe -m ruff check python/src python/tests
```

### `test_pipeline.py`

- determinism pentru același TIC/seed;
- formă scalogramă `32x512`;
- valori finite;
- normalizare în jurul medianei 1;
- respingerea datelor fără suficiente cadențe valide.

### `test_model.py`

- contractul formelor modelului;
- pozitivitatea parametrilor prin softplus;
- physics loss mai mare pentru durată imposibilă.

### `test_inference.py`

- contract `Prediction`;
- diferențierea `bls_screening_score`;
- proveniență MAST;
- propagarea sectorului;
- endpoint health/inference;
- lista produselor observaționale.

Rețeaua este înlocuită în teste prin dependency injection cu o observație sintetică deterministă. Aceasta este corect pentru unit tests și nu schimbă traseul live.

### `test_training_reporting.py`

- formele batch-ului smoke;
- existența disclaimer-ului și câmpurilor raportului.

## 4. Rust

Comenzi:

```powershell
cargo fmt --manifest-path apps/api-rust/Cargo.toml -- --check
cargo clippy --manifest-path apps/api-rust/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path apps/api-rust/Cargo.toml
```

Testele curente acoperă:

- health route;
- target search;
- distanța unghiulară pentru coordonate identice;
- propagarea `sector` în URL-ul Python.

Lipsesc încă integration tests cu un mock HTTP server pentru toate răspunsurile upstream și maparea erorilor.

## 5. Flutter

Comenzi:

```powershell
Push-Location apps/client-flutter
dart format --output=none --set-exit-if-changed lib test
flutter analyze
flutter test
flutter build web --release
Pop-Location
```

Testele curente acoperă:

- randarea workbench-ului la viewport desktop;
- parsarea observațiilor;
- includerea sectorului selectat în URL;
- parsarea predicției.

Lipsesc widget tests pentru loading/error/empty, alegerea sectorului și randarea rezultatului complet.

## 6. Testul end-to-end real

Acest test necesită internet și nu trebuie inclus automat în fiecare unit test/PR fără control asupra rate limits.

```powershell
$obs = Invoke-RestMethod `
  'http://127.0.0.1:8080/api/v1/targets/150428135/observations'

$result = Invoke-RestMethod -Method Post `
  'http://127.0.0.1:8080/api/v1/analyses/150428135?sector=3'

if ($result.prediction.sector -ne 3) { throw 'Sector mismatch' }
if ($result.prediction.data_source -notlike 'mast:TESS/product/*') {
  throw 'Missing MAST provenance'
}
if ($result.prediction.score_kind -ne 'bls_screening_score') {
  throw 'Unexpected method without checkpoint'
}
```

Nu aserta valori BLS exacte în integration test, deoarece arhiva sau versiunile bibliotecilor pot modifica ușor rezultatul. Asertă invariante și toleranțe.

## 7. CI

Workflow: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

La push și pull request rulează trei job-uri independente:

- Python 3.11: install, pytest, Ruff;
- Rust stable: rustfmt, Clippy, cargo test;
- Flutter stable: pub get, format, analyze, test, build web release.

CI nu rulează MAST live, Docker Compose, CUDA sau training real.

## 8. Testarea schimbărilor științifice

Unit tests nu sunt suficiente pentru schimbări de metodă. Pentru detrending, grilă BLS, CWT, loss sau model sunt necesare:

- benchmark pe același manifest;
- split identic;
- minimum trei seed-uri pentru modelele finale;
- metrici și intervale bootstrap;
- cost de procesare/memorie;
- analiză pe subgrupuri și false positives;
- actualizarea documentației metodologice.

## 9. Toleranțe numerice

- folosește `np.testing.assert_allclose` pentru vectori;
- declară toleranța și motivul;
- nu compara float-uri brute prin egalitate dacă provin din algoritmi numerici;
- verifică finiteness și domeniul fizic;
- verifică formele înainte de valori.

## 10. Testarea contractelor

Orice câmp nou în predicție trebuie testat cel puțin la producătorul Python și parserul Dart. Rust folosește `Value`, deci type mismatch-urile pot traversa gateway-ul și eșua doar în UI.

Prioritate viitoare: generarea DTO-urilor Flutter/Rust din OpenAPI sau JSON Schema și contract tests automate.

## 11. Definition of Done

O modificare este finalizată când:

- are test focalizat pentru comportamentul nou/reparat;
- suitele componentei trec;
- nu introduce warnings relevante;
- contractele și documentația sunt sincronizate;
- pentru schimbări cross-service, traseul end-to-end este verificat;
- pentru frontend, desktop și mobil sunt inspectate;
- pentru știință, rezultatul nu depășește dovezile disponibile;
- datele și artefactele mari nu intră în Git.

## 12. Verificarea documentației

Înainte de merge:

- verifică diagnosticele Markdown;
- verifică link-urile relative către fișiere existente;
- verifică comenzile pe sistemul țintă;
- caută nume vechi de endpoint-uri/variabile;
- compară cifrele despre teste cu rezultatul curent;
- nu păstra rezultate numerice ca adevăr canonic dacă sunt doar exemple.