# Contractele API

## 1. Convenții generale

- API public: Rust, port implicit `8080`, prefix `/api/v1`.
- API intern: Python, port implicit `8001`, prefix `/v1`.
- payload: JSON UTF-8.
- coordonate: grade ICRS.
- TIC ID și sector: numere întregi pozitive.
- analiza este sincronă, cu timeout Rust de 120 s.
- versiunea publică este v1; schimbările incompatibile cer un prefix nou sau migrare explicită.

Contractul formal este [`schemas/openapi.yaml`](../schemas/openapi.yaml). Acest document adaugă semantica pe care schema compactă nu o exprimă încă.

## 2. API public Rust

### `GET /health/live`

### `GET /health/ready`

Răspuns:

```json
{
  "status": "ok",
  "service": "api",
  "version": "0.1.0"
}
```

În MVP, live și ready folosesc același handler și nu verifică efectiv MAST sau Python.

### `GET /api/v1/targets`

Parametri opționali:

| Parametru | Tip | Reguli |
|---|---|---|
| `ra` | number | trebuie furnizat împreună cu `dec` |
| `dec` | number | trebuie furnizat împreună cu `ra` |
| `radius_deg` | number | 0..180, implicit 180 |
| `min_score` | number | implicit 0; țintele seed nu au scor preevaluat |

Răspuns:

```json
{
  "items": [
    {
      "tic_id": 150428135,
      "name": "TOI 700",
      "ra": 97.0967865903209,
      "dec": -65.5793115481931,
      "magnitude": 10.9102,
      "spectral_type": "M2V",
      "score": null
    }
  ],
  "total": 1
}
```

Catalogul curent este seed data verificată, nu o interogare dinamică TIC.

### `GET /api/v1/targets/{tic_id}`

Returnează un `Target` dacă ID-ul există în catalogul Rust; altfel `404 TARGET_NOT_FOUND`.

### `GET /api/v1/targets/{tic_id}/observations`

Interoghează MAST prin Python și returnează un produs preferat per sector.

```json
{
  "items": [
    {
      "sector": 3,
      "mission": "TESS Sector 03",
      "year": 2018,
      "exposure_seconds": 120.0,
      "source_uri": "mast:TESS/product/..._lc.fits"
    }
  ],
  "total": 37
}
```

Lista este ordonată crescător după sector. Python ține inventarul în cache de proces pentru maximum 128 ținte.

### `POST /api/v1/analyses/{tic_id}?sector={sector}`

`sector` este opțional în contract, dar clientul Flutter trimite explicit alegerea utilizatorului. Dacă lipsește, Python selectează primul produs SPOC disponibil.

Răspuns:

```json
{
  "status": "completed",
  "prediction": {
    "tic_id": 150428135,
    "probability": 0.2294,
    "label": "unlikely",
    "depth": 0.0003903,
    "duration_days": 0.3,
    "period_days": 0.5868,
    "model_version": "bls-real-0.1.0",
    "score_kind": "bls_screening_score",
    "data_source": "mast:TESS/product/..._lc.fits",
    "mission": "TESS/SPOC",
    "sector": 3,
    "cadence_count": 13541,
    "time": [1385.936, 1386.095],
    "normalized_flux": [1.0001, 0.9998]
  }
}
```

Valorile din exemplu sunt orientative pentru contract, nu rezultate canonice.

## 3. API intern Python

### `GET /health`

```json
{"status":"ok","service":"inference"}
```

### `GET /v1/observations/{tic_id}`

Returnează același `ObservationProductList` pe care Rust îl proxiază public.

### `POST /v1/infer/{tic_id}?sector={sector}`

Rulează download, preprocesare și BLS/model. Nu trebuie apelat direct de clientul public într-un deployment securizat.

## 4. DTO-uri

### `Target`

| Câmp | Tip | Semnificație |
|---|---|---|
| `tic_id` | integer | identificator TIC |
| `name` | string | nume afișabil |
| `ra` | number | ascensie dreaptă ICRS, grade |
| `dec` | number | declinație ICRS, grade |
| `magnitude` | number | TESS magnitude pentru seed catalog |
| `spectral_type` | string | tip cunoscut sau `TIC star` |
| `score` | number/null | scor preevaluat; momentan null |

### `ObservationProduct`

| Câmp | Tip | Semnificație |
|---|---|---|
| `sector` | integer | sectorul TESS |
| `mission` | string | eticheta produsului din MAST |
| `year` | integer | anul observației |
| `exposure_seconds` | number | timpul de expunere/cadența produsului |
| `source_uri` | string | URI-ul MAST al FITS-ului |

### `Prediction`

| Câmp | Tip | Unitate/semnificație |
|---|---|---|
| `probability` | number | scor 0..1; interpretează prin `score_kind` |
| `label` | string | `candidate` sau `unlikely` |
| `depth` | number | flux relativ $\Delta F/F$ |
| `duration_days` | number | zile |
| `period_days` | number | zile |
| `model_version` | string | versiune BLS sau nume checkpoint |
| `score_kind` | string | BLS screening ori AI nevalidat |
| `data_source` | string | URI MAST |
| `mission` | string | `TESS/SPOC` |
| `sector` | integer | sectorul efectiv analizat |
| `cadence_count` | integer | punctele curbei după download/mask Lightkurve |
| `time` | number[] | seria downsampled pentru UI |
| `normalized_flux` | number[] | fluxul normalizat, aceeași lungime ca `time` |

## 5. Erori publice

Format Rust:

```json
{
  "code": "INVALID_TIC_ID",
  "message": "tic_id must be positive"
}
```

Coduri curente:

| Cod | HTTP | Cauză |
|---|---:|---|
| `INVALID_COORDINATES` | 422 | RA/Dec incomplet |
| `INVALID_RADIUS` | 422 | raza în afara 0..180 |
| `INVALID_TIC_ID` | 422 | ID zero |
| `TARGET_NOT_FOUND` | 404 | ținta nu este în catalogul public |
| `MAST_UNAVAILABLE` | 503 | serviciul de observații nu răspunde |
| `OBSERVATION_QUERY_FAILED` | 502 | Python/MAST a returnat eroare |
| `INVALID_OBSERVATION_RESPONSE` | 502 | JSON intern invalid |
| `INFERENCE_UNAVAILABLE` | 503 | Python nu poate fi contactat |
| `INFERENCE_FAILED` | 502 | inferența internă a eșuat |
| `INVALID_INFERENCE_RESPONSE` | 502 | predicție JSON invalidă |

Python folosește `detail` FastAPI pentru erorile interne. Rust nu expune momentan corpul detaliat al erorii upstream.

## 6. Exemple PowerShell

```powershell
# health
Invoke-RestMethod http://127.0.0.1:8080/health/ready

# catalog
Invoke-RestMethod 'http://127.0.0.1:8080/api/v1/targets?min_score=0'

# sectoare reale
Invoke-RestMethod 'http://127.0.0.1:8080/api/v1/targets/150428135/observations'

# analiză explicită sector 3
Invoke-RestMethod -Method Post `
  'http://127.0.0.1:8080/api/v1/analyses/150428135?sector=3'
```

## 7. Reguli pentru schimbarea contractului

La adăugarea sau schimbarea unui câmp:

1. modifică producătorul Python dacă este rezultat științific;
2. actualizează proxy-ul/validarea Rust;
3. actualizează `models.dart` și `api_client.dart`;
4. actualizează OpenAPI și acest document;
5. adaugă teste Python, Rust și/sau Dart pentru propagare;
6. rulează un test end-to-end cu un sector MAST real.