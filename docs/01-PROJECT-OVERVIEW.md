# Prezentarea proiectului

## 1. Identitate și scop

**AstroMap-AI** este o platformă software de cercetare pentru descărcarea, procesarea și inspecția curbelor de lumină TESS în vederea identificării semnalelor compatibile cu tranzite exoplanetare.

Proiectul urmărește două rezultate distincte:

1. un experiment științific reproductibil care compară BLS și o rețea neurală dual-domain cu regularizare fizică;
2. un instrument interactiv care permite alegerea unei stele și a unui sector TESS, rularea analizei și inspectarea rezultatului împreună cu proveniența datelor.

Platforma este software de screening. Ea nu confirmă exoplanete și nu înlocuiește validarea astrofizică, observațiile de follow-up sau evaluarea unui astronom.

## 2. Starea implementării

| Capacitate | Stare | Observații |
|---|---|---|
| Catalog ținte | MVP | Trei ținte TIC verificate, păstrate în Rust |
| Inventar observații | Funcțional | Produse SPOC interogate din MAST și deduplicate pe sector |
| Descărcare FITS | Funcțional | Lightkurve, cu cache local |
| Preprocesare | Funcțional | Quality mask, detrending, normalizare, resampling și scalogramă |
| Baseline BLS | Funcțional | Mod implicit pentru analiza live |
| Rețea PyTorch | Implementată | Arhitectură dual-domain și încărcare checkpoint |
| Training pe date reale | Neimplementat | Runner-ul existent este smoke training sintetic |
| Validare științifică AI | Neimplementată | Nu există checkpoint aprobat pentru rezultate de teză |
| API public Rust | Funcțional | HTTP sincron, timeout 120 s |
| Client Flutter | Funcțional | Web/Windows, selector țintă și sector |
| Rapoarte | MVP | HTML imprimabil |
| Docker Compose | Pregătit | Nu a fost validat în mediul Windows curent |
| Persistență/queue | Neimplementat | Fără PostgreSQL, Redis sau job workers |

## 3. Fluxul utilizatorului

1. Utilizatorul deschide aplicația Flutter.
2. Clientul încarcă țintele din gateway-ul Rust.
3. La selectarea unei ținte, gateway-ul cere serviciului Python inventarul SPOC din MAST.
4. Utilizatorul alege sectorul, anul și expunerea dorite.
5. Comanda de analiză traversează Flutter -> Rust -> Python.
6. Python descarcă produsul FITS sau îl reutilizează din cache.
7. Curba este curățată, detrenduită, normalizată și transformată.
8. Fără checkpoint, Astropy BLS produce perioada, durata, adâncimea și scorul de screening.
9. Cu checkpoint configurat, `PinnTransNet` produce un scor și trei parametri pozitivi; rezultatul rămâne etichetat nevalidat până la evaluarea formală.
10. Flutter afișează rezultatul, curba normalizată, metoda, sectorul, numărul de cadențe și URI-ul MAST.

## 4. Tehnologii

### Python 3.11+

- Lightkurve și Astropy pentru MAST/FITS/BLS;
- NumPy și SciPy pentru procesare numerică;
- PyTorch pentru model și training;
- FastAPI/Uvicorn pentru serviciul intern;
- pytest și Ruff pentru calitate.

### Rust

- Axum și Tokio pentru API;
- Reqwest pentru comunicarea cu Python;
- Serde pentru contracte JSON;
- Tower HTTP pentru CORS și tracing.

### Flutter/Dart

- Flutter Web și Windows;
- `package:http` pentru API;
- `CustomPainter` pentru proiecția cerească și curba de lumină;
- teste widget și contract HTTP.

## 5. Terminologie

| Termen | Sens în proiect |
|---|---|
| TIC | TESS Input Catalog identifier |
| Sector | Interval/câmp de observație TESS |
| SPOC | Science Processing Operations Center, sursa produselor analizate |
| Cadență | Un eșantion fotometric la un anumit timp |
| BTJD | Barycentric TESS Julian Date; timpul furnizat de curbele TESS |
| BLS | Box Least Squares, baseline clasic pentru tranzite periodice |
| CWT/scalogramă | Reprezentare timp-scală calculată cu wavelet Ricker |
| Physics loss | Penalizare pentru durate incompatibile cu o orbită circulară centrală |
| Screening score | Scor operațional, nu probabilitate calibrată și nu confirmare |
| Checkpoint | Starea parametrilor PyTorch salvată într-un fișier `.pt` |

## 6. Structura repository-ului

```text
Teza_Master/
├── .github/                     # CI și instrucțiuni Copilot
├── apps/
│   ├── api-rust/                # gateway public
│   └── client-flutter/          # UI web/desktop
├── configs/                     # configurații de experiment
├── data/                        # manifeste și fixtures; datele mari nu intră în Git
├── docs/                        # documentația curentă
├── infra/                       # Dockerfiles, Nginx și Compose
├── python/                      # pachetul științific și testele sale
├── schemas/                     # OpenAPI
├── scripts/                     # automatizare locală
├── artifacts/                   # checkpoint-uri/rapoarte locale, ignorate de Git
├── README.md
├── PLAN_COMPLET_ASTROMAP_AI.md
└── specifica_ie_proiect_master_ai_exoplanete.md
```

## 7. Surse de adevăr

- comportament runtime: codul și testele curente;
- API public: rutele Rust plus [`schemas/openapi.yaml`](../schemas/openapi.yaml);
- contract Python: dataclass-urile `ObservationProduct`, `RealObservation` și `Prediction`;
- metodologie implementată: [`pipeline.py`](../python/src/astromap/pipeline.py), [`model.py`](../python/src/astromap/model.py) și [`inference.py`](../python/src/astromap/inference.py);
- viziune și etape viitoare: [`PLAN_COMPLET_ASTROMAP_AI.md`](../PLAN_COMPLET_ASTROMAP_AI.md).

## 8. Ce nu trebuie presupus

- existența unui checkpoint real doar pentru că arhitectura modelului există;
- faptul că un scor mare BLS confirmă o planetă;
- faptul că perioada găsită într-un singur sector este perioada orbitală adevărată;
- faptul că modelul actual conține TCN sau attention; acestea sunt propuneri din roadmap;
- faptul că Docker, CUDA ori deployment-ul Vast.ai au fost validate doar pentru că există fișiere de configurare;
- faptul că cele trei ținte seed reprezintă datasetul de cercetare.

## 9. Criteriul de finalizare a tezei

MVP-ul software nu este echivalent cu finalizarea cercetării. Rezultatul științific necesită:

- catalog etichetat și versionat;
- split grupat pe stea, fără leakage;
- baseline-uri și ablații pe același test blocat;
- checkpoint-uri și configurații trasabile;
- metrici cu intervale de încredere și calibrare;
- analiză a erorilor și limitărilor;
- reproducerea automată a tabelelor și figurilor tezei.