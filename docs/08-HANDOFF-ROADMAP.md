# Handoff și roadmap

## 1. Rezumat pentru următorul developer/agent

Proiectul este un MVP integrat și testat local. Fluxul funcțional este:

```text
Flutter -> Rust -> Python -> NASA MAST -> preprocessing -> BLS -> UI
```

TOI 700 (`TIC 150428135`) are inventar real de sectoare, iar sectorul ales este propagat end-to-end. BLS este metoda implicită. Modelul PyTorch poate fi încărcat prin env, dar nu există un checkpoint real validat care să fie promovat.

Începe orice sesiune astfel:

1. citește [instrucțiunile workspace](../.github/copilot-instructions.md);
2. citește [overview-ul](01-PROJECT-OVERVIEW.md) și documentul componentei;
3. verifică modificările necomise și nu le suprascrie;
4. rulează testele baseline pentru componenta atinsă;
5. formulează o ipoteză locală înainte de prima editare.

## 2. Ce este stabil

- separarea Python/Rust/Flutter;
- contractul de inventariere a sectoarelor;
- proveniența MAST în `Prediction`;
- preprocesarea fixă 512 și scalograma 32x512;
- fallback-ul explicit la BLS;
- selectorul de sector Flutter;
- testele unitare și build-ul Flutter Web;
- structura Docker multi-service.

„Stabil” înseamnă că schimbările trebuie motivate și testate, nu că designul este definitiv.

## 3. Limitări și datorii cunoscute

### P0 - cercetare

1. Nu există dataset real etichetat și versionat.
2. Nu există split blocat pe TIC ID.
3. Nu există checkpoint AI validat.
4. Nu există benchmark complet BLS versus model.
5. Nu există studiu de ablație CWT/physics loss.

Acestea blochează afirmațiile centrale ale tezei.

### P1 - corectitudine și reproducibilitate

1. `configs/training.yaml` nu este consumat integral de runner-ul smoke.
2. Pipeline-ul nu salvează un manifest al produselor și checksum-urilor.
3. Checkpoint-ul nu are model card/checksum/config atașat.
4. Readiness Rust nu verifică Python.
5. Nu există versionare explicită a pipeline-ului în predicție.
6. `score_kind=unvalidated_ai_probability` nu este calibrat.

### P2 - operare

1. Analiza este sincronă și ține conexiunea HTTP deschisă.
2. Nu există Redis/job queue sau anulare.
3. Nu există PostgreSQL/PostGIS ori istoric de predicții.
4. Cache-ul MAST nu este volum persistent în Compose.
5. CORS este permisiv.
6. Docker nu a fost validat local.
7. Lipsesc TLS, rate limiting și observabilitate structurată.

### P2 - frontend

1. Catalogul Rust are doar trei ținte statice.
2. Sky map-ul este o proiecție custom, nu Aladin Lite/HiPS.
3. Nu există phase-folded plot sau scalogramă în UI.
4. Nu există export CSV/PNG ori raport din UI.
5. `apps/client-flutter/lib/main.dart` conține încă clasele demo Flutter nefolosite.
6. Lipsesc teste UI pentru toate stările.

### P3 - metodă

1. Nu există TCN/attention/TTV, deși apar în viziunea inițială.
2. Nu există Gaussian Process detrending.
3. Nu există uncertainty estimation.
4. Nu există analiză multi-sector ori cross-mission Kepler/TESS.
5. Physics loss nu este Mandel-Agol complet.

## 4. Roadmap recomandat

### Etapa A - dataset real reproducibil

Livrabile:

- script de import NASA Exoplanet Archive;
- cross-match TIC/Gaia;
- reguli de etichetare și conflict;
- manifest Parquet/JSON cu checksum;
- split train/validation/test grupat pe TIC;
- raport de distribuții și audit manual.

Gate: zero TIC comun între partiții.

### Etapa B - baseline științific

Livrabile:

- BLS pe întregul manifest;
- BLS + model clasic;
- metrici PR-AUC, ROC-AUC, FPR@95% recall;
- calibrare și intervale bootstrap;
- error taxonomy.

Gate: cifre reproductibile dintr-o comandă.

### Etapa C - training neural real

Livrabile:

- Dataset/DataLoader pe manifest;
- configurație unică versionată;
- mixed precision și checkpointing;
- minimum trei seed-uri;
- model card și checksum;
- evaluare pe validation fără acces la test.

Gate: checkpoint-ul poate fi încărcat de `InferenceEngine` și reproduce metricile.

### Etapa D - ablații și test final

Experimente:

- temporal-only;
- temporal + wavelet;
- fără physics loss;
- cu physics loss;
- comparație BLS;
- test sintetic TTV separat de testul real.

Gate: configurația finală este înghețată înaintea testului blocat.

### Etapa E - platformă operațională

- job queue și status polling/SSE;
- DB și object storage;
- Aladin Lite;
- grafice phase-folded/scalogramă;
- raport din UI;
- Docker/GPU deployment, TLS și monitoring.

## 5. Ordinea recomandată a următoarelor task-uri

1. elimină boilerplate-ul nefolosit din `main.dart`, cu test Flutter;
2. adaugă integration tests Rust cu mock downstream;
3. creează schema manifestului de dataset;
4. implementează importerul de catalog etichetat;
5. implementează split-ul grupat și testul anti-leakage;
6. salvează produsele preprocesate versionat;
7. construiește runner-ul real de training;
8. adaugă evaluarea și model card-ul;
9. numai apoi activează checkpoint-ul în deployment.

Primele două sunt cleanup tehnic. Task-urile 3-8 sunt traseul critic al tezei.

## 6. Decizii care nu trebuie inversate fără ADR

- Python rămâne proprietarul logicii științifice.
- Rust rămâne gateway, nu al doilea motor de inferență.
- Flutter nu primește FITS brute și nu calculează BLS.
- traseul live nu folosește date sintetice.
- toate sectoarele aceleiași stele aparțin aceluiași split.
- BLS rămâne baseline și fallback transparent.
- checkpoint-urile sunt artefacte imuabile, nu fișiere suprascrise manual.
- scorul și metoda sa sunt afișate împreună.

Pentru o schimbare majoră, adaugă un document în `docs/adr/NNNN-titlu.md` cu context, decizie, alternative și consecințe.

## 7. Anti-patterns

- hardcodarea unui rezultat cunoscut pentru un TIC;
- returnarea unei curbe sintetice când MAST eșuează;
- split aleator pe ferestre/cadențe;
- selectarea hiperparametrilor pe test set;
- numirea unui sigmoid necalibrat „confidence”;
- salvarea datelor FITS/checkpoint-urilor în Git;
- dublarea DTO-ului fără test de contract;
- creșterea timeout-ului în locul unei cozi de joburi;
- refactorizarea simultană a pipeline-ului și modelului fără ablație.

## 8. Template de handoff pentru o sesiune

```markdown
## Obiectiv
Ce comportament s-a urmărit.

## Modificări
Fișiere și contracte schimbate.

## Dovezi
Comenzi/teste și rezultate relevante.

## Date/artefacte
Manifest, checkpoint, checksum, config și locație.

## Limitări
Ce nu a fost validat sau implementat.

## Următorul pas
Cel mai mic task care continuă în siguranță.
```

## 9. Checklist înainte de predare

- worktree-ul și modificările utilizatorului au fost respectate;
- toate testele relevante au fost rulate;
- serverele de dezvoltare rămase sunt identificate;
- documentația și OpenAPI sunt sincronizate;
- nu există date sau secrete adăugate accidental;
- afirmațiile științifice sunt proporționale cu dovezile;
- orice rezultat numeric important este legat de manifest/checkpoint;
- blocajele și pașii neexecutați sunt expliciți.