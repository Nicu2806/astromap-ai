# Deployment și operare

## 1. Topologia disponibilă

Fișierul [`infra/compose/docker-compose.yml`](../infra/compose/docker-compose.yml) definește:

| Serviciu | Imagine | Port host | Rol |
|---|---|---:|---|
| `inference` | Python 3.11 slim | 8001 | MAST, pipeline, BLS/PyTorch |
| `api` | Rust Debian slim | 8080 | gateway public |
| `client` | Flutter build + Nginx | 8088 | UI static |

Compose este pregătit, dar nu a fost executat în mediul Windows în care s-a realizat MVP-ul deoarece Docker nu era instalat. Prima rulare pe Linux/Vast.ai trebuie tratată ca validare de deployment, nu ca operație deja certificată.

## 2. Build și pornire

```powershell
docker compose -f infra/compose/docker-compose.yml build
docker compose -f infra/compose/docker-compose.yml up -d
docker compose -f infra/compose/docker-compose.yml ps
```

Health checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
Invoke-RestMethod http://127.0.0.1:8080/health/ready
Invoke-WebRequest http://127.0.0.1:8088 -UseBasicParsing
```

Test funcțional:

```powershell
$observations = Invoke-RestMethod `
  http://127.0.0.1:8080/api/v1/targets/150428135/observations

$result = Invoke-RestMethod -Method Post `
  'http://127.0.0.1:8080/api/v1/analyses/150428135?sector=3'
```

Verifică `sector`, `cadence_count`, `data_source` și `score_kind`, nu doar HTTP 200.

## 3. Variabile de mediu

| Variabilă | Componentă | Implicit | Descriere |
|---|---|---|---|
| `ASTROMAP_CHECKPOINT` | Python | gol | cale `.pt`; trebuie să existe |
| `ASTROMAP_INFERENCE_URL` | Rust | `http://127.0.0.1:8001` | URL intern Python |
| `ASTROMAP_API_PORT` | Rust | `8080` | port listen |
| `RUST_LOG` | Rust | fără filtru explicit | nivel tracing |
| `ASTROMAP_API_URL` | Flutter build | `http://127.0.0.1:8080` | URL public compile-time |
| `PYTHONDONTWRITEBYTECODE` | Python image | `1` | fără `.pyc` |
| `PYTHONUNBUFFERED` | Python image | `1` | loguri imediate |

În Compose, `ASTROMAP_CHECKPOINT` este preluat din mediul host, iar `artifacts/` este montat la `/app/artifacts` read-only. Valoarea trebuie să fie o cale validă **în container**, de exemplu `/app/artifacts/model.pt`, nu calea Windows de pe host.

## 4. Deployment pe server GPU/Vast.ai

GPU-ul este necesar pentru training și opțional pentru inferența PyTorch. BLS funcționează pe CPU.

Pași recomandați:

1. alege o imagine/mașină cu CUDA compatibilă cu PyTorch-ul instalat;
2. montează un volum persistent separat pentru dataset și `artifacts/`;
3. păstrează serviciile web în folderul repository-ului și datele mari în volume;
4. verifică `torch.cuda.is_available()` în worker-ul de training;
5. antrenează separat de API;
6. copiază numai checkpoint-ul promovat și manifestul său în `artifacts/`;
7. repornește inference cu `ASTROMAP_CHECKPOINT` explicit;
8. rulează smoke API și un benchmark de latență.

Imaginea Python curentă este CPU-oriented și nu configurează NVIDIA Container Toolkit. Pentru inferență GPU trebuie introdusă o imagine CUDA/PyTorch separată și validată.

## 5. Persistență și cache

### Situația actuală

- `artifacts/` este volum read-only;
- cache-ul Lightkurve rămâne în filesystem-ul containerului;
- restart simplu păstrează containerul, recrearea lui pierde cache-ul;
- nu există DB pentru predicții sau audit history.

### Recomandare

Adaugă un volum dedicat pentru home/cache Python, de exemplu:

```yaml
volumes:
  - mast-cache:/home/astromap/.lightkurve
```

Confirmă locația efectivă a cache-ului pentru userul non-root din imagine înainte de configurare.

## 6. Reverse proxy și TLS

Nginx-ul curent servește doar build-ul Flutter pe HTTP. Pentru expunere publică:

- termină TLS prin Caddy/Nginx/load balancer;
- expune numai UI și Rust API;
- nu expune direct portul 8001;
- limitează CORS la origin-ul aplicației;
- adaugă rate limiting pentru endpoint-ul de analiză;
- setează limite de request și timeout-uri;
- adaugă headers de securitate;
- nu include secrete în imagini sau în Git.

## 7. Observabilitate

### Loguri curente

- Uvicorn: request line, status și erori Python;
- Rust: tracing controlat prin `RUST_LOG`;
- Nginx: access/error logs standard.

### Cerințe pentru producție

- loguri JSON structurate;
- `trace_id` propagat Rust -> Python;
- metrici pentru latență MAST, cache hit, preprocessing, BLS/model și erori;
- alerte pentru creșterea 5xx și timeouts;
- fără fluxuri fotometrice complete în loguri.

## 8. Readiness corect

`/health/ready` Rust verifică momentan doar procesul Rust. Pentru producție trebuie să verifice cel puțin:

- accesul la serviciul Python;
- încărcarea checkpoint-ului configurat;
- disponibilitatea storage-ului necesar.

MAST nu trebuie neapărat inclus în readiness, deoarece o indisponibilitate externă temporară nu înseamnă că serviciul local trebuie scos din rotație; ea trebuie reflectată în metrici și răspunsul operației.

## 9. Runbook pentru incidente

### API 503 `INFERENCE_UNAVAILABLE`

1. verifică procesul/containerul Python;
2. apelează `/health` direct în rețeaua internă;
3. verifică `ASTROMAP_INFERENCE_URL`;
4. inspectează logurile Uvicorn/importurile;
5. verifică existența checkpoint-ului configurat.

### API 502 `OBSERVATION_QUERY_FAILED`

1. verifică logul Python pentru mesaj MAST;
2. testează conectivitatea DNS/HTTPS spre MAST;
3. verifică TIC ID și existența produselor SPOC;
4. nu înlocui automat datele cu fixture sintetic.

### Latență mare

1. separă timpul query MAST de download și procesare;
2. verifică cache hit;
3. monitorizează dimensiunea produsului/cadența;
4. limitează concurența pentru a proteja serviciul extern;
5. implementează job queue înainte de a extinde timeout-ul excesiv.

### Rezultat suspect

1. verifică `score_kind`;
2. verifică sectorul și baseline-ul temporal;
3. inspectează curba și quality mask;
4. compară cu BLS direct;
5. nu modifica manual rezultatul; deschide un caz reproductibil.

## 10. Release checklist

- toate testele și lint-urile trec;
- OpenAPI și documentația sunt actualizate;
- imaginile sunt build-uite din commit-ul etichetat;
- checkpoint-ul are checksum, config și manifest;
- `score_kind` este corect;
- endpoint-ul observations și o analiză reală trec;
- UI desktop și mobil sunt inspectate;
- backup/rollback este definit;
- TLS, CORS și rate limits sunt verificate;
- disclaimer-ul științific este vizibil.

## 11. Rollback

Păstrează imaginile versionate prin commit SHA și checkpoint-urile imuabile. Rollback-ul trebuie să promoveze împreună:

- imaginea Python compatibilă;
- checkpoint-ul și config-ul lui;
- imaginea Rust compatibilă cu contractul;
- build-ul Flutter compatibil cu DTO-urile.

Nu combina un client nou cu un API vechi fără contract tests.