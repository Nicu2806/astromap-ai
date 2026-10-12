# Documentație AstroMap-AI

Acest folder este sursa principală de documentație tehnică și științifică pentru AstroMap-AI. Documentele descriu implementarea existentă la 1 octombrie 2026, nu doar arhitectura propusă în specificația inițială.

## Starea proiectului

AstroMap-AI este un MVP funcțional cu trei componente:

- serviciu științific Python pentru NASA MAST, preprocesare, BLS și modelul PyTorch;
- gateway HTTP Rust/Axum;
- aplicație Flutter Web/Windows pentru alegerea țintei, sectorului și inspecția rezultatului.

Analiza live utilizează curbe TESS/SPOC reale. În lipsa unui checkpoint validat, rezultatul este un **scor BLS de screening**, nu o probabilitate AI și nu confirmă existența unei exoplanete. Training-ul inclus este un smoke test sintetic, nu rezultatul științific final al tezei.

## Ordine recomandată de lectură

1. [Prezentarea proiectului](01-PROJECT-OVERVIEW.md) - scop, stare, terminologie și limite.
2. [Arhitectura sistemului](02-ARCHITECTURE.md) - componente, fluxuri și responsabilități.
3. [Metodologia științifică](03-SCIENTIFIC-METHODOLOGY.md) - date, preprocesare, BLS, model și validare.
4. [Contractele API](04-API-CONTRACTS.md) - endpoint-uri, DTO-uri și erori.
5. [Ghidul de dezvoltare](05-DEVELOPMENT-GUIDE.md) - instalare, rulare și convenții.
6. [Deployment și operare](06-DEPLOYMENT-OPERATIONS.md) - Docker, configurare, cache și runbook.
7. [Testare și calitate](07-TESTING-QUALITY.md) - testele obligatorii și verificările CI.
8. [Handoff și roadmap](08-HANDOFF-ROADMAP.md) - starea curentă, datorii și următorii pași.

## Documente de origine

- [Specificația inițială](../specifica_ie_proiect_master_ai_exoplanete.md) exprimă viziunea de cercetare.
- [Planul complet](../PLAN_COMPLET_ASTROMAP_AI.md) descrie roadmap-ul extins și criteriile tezei.
- [README-ul principal](../README.md) conține comenzile rapide pentru utilizare locală.
- [OpenAPI](../schemas/openapi.yaml) este contractul formal al API-ului public.

În caz de contradicție, comportamentul verificat prin teste și codul curent au prioritate față de planurile istorice. Contradicția trebuie apoi corectată în documentație.

## Pentru agenți AI

Instrucțiunile permanente sunt în [`.github/copilot-instructions.md`](../.github/copilot-instructions.md). Înainte de o modificare, agentul trebuie să consulte documentul relevant din acest folder și testele componentei afectate.

## Actualizarea documentației

Actualizează documentația în aceeași schimbare atunci când se modifică:

- un endpoint sau un DTO;
- o variabilă de mediu, un port sau o comandă;
- preprocesarea, BLS, arhitectura modelului sau loss-ul fizic;
- semnificația unui scor ori a unei etichete;
- structura repository-ului, deployment-ul sau procedura de testare.

Nu documenta rezultate științifice înainte ca acestea să fie reproduse dintr-un manifest de date versionat și un checkpoint identificabil.