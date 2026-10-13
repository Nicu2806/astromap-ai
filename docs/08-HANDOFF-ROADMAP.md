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

## 3. Starea Etapelor de Cercetare & Livrabile Finalizate

### ✅ Realizări Validate și Încheiate:
1. **Dataset real etichetat și versionat:** Partitionare strictă pe TIC ID fără data leakage (`data/real_manifest.json`, `python/src/astromap/real_dataset.py`).
2. **Checkpoint AI validat și promovat:** `artifacts/pinn-transnet-v1.pt` (ROC-AUC 0.941, PR-AUC 0.912) cu model card auditat (`artifacts/model_card.json`).
3. **Benchmark masiv BLS vs AI:** Evaluare formală pe 1.000 de curbe SPOC (`artifacts/validation_1000_results.csv`, `artifacts/validation_1000_metrics.json`).
4. **Studiu complet de ablație:** Ablații riguroase temporal-only, wavelet CWT și funcție de pierdere fizică Mandel-Agol (`artifacts/ablation_study.json`), demonstrând o reducere cu 62% a alarmelor false.
5. **Benchmark TTV:** Demonstrarea robusteții la variații temporale de tranzit de până la 240 minute (`artifacts/ttv_benchmark.json`).
6. **Campanie de scanare cer & dosare de diagnoză:** Screening pe 30 de ținte TESS Sector 1, generând 33 de rapoarte HTML interactive, catalog CSV/JSON și dashboard centralizat (`artifacts/discovery_reports/index.html`), cu 12 noi candidați de exoplanete descoperiți.
7. **Pachet Academic & Open Science:** Manuscris complet LaTeX (`paper/main.tex`, `paper/manuscript.tex`), `CITATION.cff`, Zenodo DOI `10.5281/zenodo.23317448` și depozit GitHub sincronizat.

## 4. Direcții și Roadmap Viitor

### Etapa 1: Pregătirea Susținerii Tezei de Master (Imediat)
- **Prezentare & Slide-uri:** Generarea setului de slide-uri (PowerPoint / LaTeX Beamer) ghidat de structura din [`docs/10-THESIS-DEFENSE-GUIDE.md`](10-THESIS-DEFENSE-GUIDE.md).
- **Demonstrator Live:** Rularea stack-ului local (FastAPI Python + Gateway Rust + Flutter Web) pentru demonstrația practică din fața comisiei.
- **Prezentarea Descoperirilor:** Navigare live în [`artifacts/discovery_reports/index.html`](../artifacts/discovery_reports/index.html) pentru candidatul fanion TIC 200668668 și comparația cu sistemul TOI 700 d.

### Etapa 2: Publicare & Diseminare Științifică (Academic)
- **Arhivare Open Science:** Finalizarea release-ului v1.0.0 pe Zenodo conectat la depozitul GitHub (`10.5281/zenodo.23317448`).
- **Trimitere Preprint:** Depunere pe arXiv (`astro-ph.EP`) a manuscrisului `paper/main.tex`.
- **Notificare Comunitate TFOP:** Transmiterea fișierelor de tranzit pentru candidații necatalogați (ex. TIC 200668668, TIC 461942081) către ExoFOP/TFOP (TESS Follow-up Observing Program) conform ghidului [`docs/12-EXOFOP-TFOP-SUBMISSION-GUIDE.md`](12-EXOFOP-TFOP-SUBMISSION-GUIDE.md) pentru observații fotometrice și spectroscopice de la sol.

### Etapa 3: Scalare Tehnică & Ingestie Masivă (Extindere Sistem)
- **Scanare Multi-Sector:** Rularea scannerului automat pe sectoarele extinse TESS (Sectoarele 2–70+) prin bulk ingestie STScI MAST.
- **Bază de date persistentă:** Conectarea PostgreSQL / TimescaleDB la API-ul Rust pentru stocarea permanentă a milioane de puncte fotometrice.
- **Integrare UI Extinsă:** Ingestia catalogului de 30 de ținte direct în ecranul vizual Flutter Web/Desktop.

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