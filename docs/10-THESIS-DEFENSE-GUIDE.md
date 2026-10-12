# Ghid de Susținere a Tezei de Master (Thesis Defense Guide)
## AstroMap-AI: Physics-Informed Interactive Exoplanet Detection System

> **Destinație:** Pregătirea completă pentru susținerea publică a tezei de master în fața comisiei academice.  
> **Timp recomandat prezentare:** 10 – 15 minute expunere + 5 minute sesiune de întrebări și răspunsuri (Q&A).

---

## 1. Structura Recomandată a Prezentării (Slide-by-Slide)

### Slide 1: Titlu și Context
* **Titlu:** *AstroMap-AI: Sistem Hibrid Physics-Informed și Arhitectură Spațială Interactivă pentru Detecția Tranzitelor Exoplanetare*
* **Autor & Coordonator Științific**
* **Domeniu:** Inteligență Artificială Aplicată, Astrofizică Computațională, Inginerie Software.

### Slide 2: Problema Științifică Deschizătoare (The Motivation)
* **Punctul slab al astronomiei clasice:** Telescoapele NASA (Kepler, TESS) folosesc algoritmul Box-Least-Squares (BLS), care presupune orbite strict circulare și ceasornice perfect rigide.
* **Unghiul mort (TTV):** În sistemele cu mai multe planete (cum e TRAPPIST-1 sau TOI-700), atracția gravitațională reciprocă provoacă *Transit Timing Variations* (decalaje de 10–240 min). BLS pierde semnalul prin pliere, iar planeta este ratată.
* **Punctul slab al AI-ului generic:** Rețelele clasice (ex. AstroNet de la Google) sunt „cutii negre” needucate fizic, care depind de faza BLS și produc alarme false pe stele binare.

### Slide 3: Obiectivele Tezei de Master
1. **Dezvoltarea unei arhitecturi dual-domain:** 1D flux temporal + 2D scalograme Continuous Wavelet Transform (CWT).
2. **Formularea unei funcții de pierdere bazate pe fizică (PINN):** Constrângerea rețelei cu modelul analitic de tranzit Mandel & Agol (2002) cu autograd.
3. **Consens Hibrid & Vetting Automat (Robovetter):** Fuziunea dintre AI, Fourier și testele eliminatorii astrofizice (Odd-Even, Secundar, Gaia DR3).
4. **Platformă completă de producție:** Pipeline Python, Gateway Rust de mare viteză și Workbench vizual Flutter.

### Slide 4: Pipeline-ul Fotometric & Transformata Undișoară (CWT)
* Ingestie directă a produselor SPOC reale din arhiva NASA MAST.
* Filtrare anomalii instrumentale și detrending Savitzky-Golay.
* Calculul scalogramei Ricker CWT pe 32 de scale: transformă căderea de lumină într-o semnătură 2D ce deosebește forma de „U” (planetă) de forma ascuțită de „V” (binară).

### Slide 5: Arhitectura Neurală PINN-TransNet
* **Encoder:** Convoluții 1D multi-scală + Temporal Convolutional Networks (TCN).
* **Mecanism de Atenție:** Multi-Head Temporal Self-Attention pentru captarea dependențelor non-locale (rezistență la TTV).
* **Dual-Head Output:** Clasificare probabilistică $P(\text{planet})$ + Regresie parametrică orbitală $(\delta, T_{\text{dur}}, P)$.

### Slide 6: Fizica Diferențiabilă Mandel & Agol (2002)
* Ecuația de întunecare quadratică spre margine:
  $$I(\mu) = 1 - u_1(1 - \mu) - u_2(1 - \mu)^2$$
* Geometrie Kepleriană integrată în graful autograd PyTorch.
* Funcția de pierdere compusă:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \lambda_{\text{phys}} \cdot \mathcal{L}_{\text{Mandel-Agol}}$$

### Slide 7: Sistemul Hibrid de Consens și Robovetter
* Evaluare sincronă PINN-TransNet + Box-Least-Squares.
* Rezonanță armonică orbitală ($1:1, 1:2, 2:1, 1:3$).
* **Sita cu 3 filtre Robovetter:**
  * Testul de Paritate Par-Impar (Odd-Even $<3.5\sigma$) $\to$ elimină stelele binare.
  * Scanarea Eclipsei Secundare ($\phi \approx 0.5$) $\to$ elimină companionii luminoși.
  * Densitatea Stelară Kepler-3 vs. Satelitul Gaia DR3 $\to$ verifică plauzibilitatea fizică.

### Slide 8: Rezultate Experimentale – Ablație și Validare (RQ1 & RQ2)
* Demonstrarea eficienței penalizării fizice (`artifacts/ablation_study.json`):
  * Modelul standard (doar BCE): PR-AUC = 0.884, FPR95 = 0.128.
  * Modelul PINN (cu $\lambda_{\text{phys}}$): PR-AUC = 0.941, FPR95 = 0.048 (**reducere cu 62% a alarmelor false**).

### Slide 9: Rezultate Experimentale – Benchmark TTV (RQ3)
* Demonstrarea superiorității față de BLS (`artifacts/ttv_benchmark.json`):
  * Pe măsură ce jitterul TTV crește de la $0$ la $240\text{ min}$, BLS pierde peste 68% din putere.
  * PINN-TransNet își menține Recall-ul la 1.00 (100%), dovedind rezolvarea unghiului mort al metodelor clasice.

### Slide 10: Arhitectura de Sistem Enterprise
* **Python:** Nucleu matematic și stocare columnar Parquet (Zstd).
* **Rust (Axum):** Gateway asincron de mare viteză cu contract OpenAPI 3.1.
* **Flutter:** Workbench cu proiecție celestă ICRS și carduri Robovetter.
* **36 de teste automate (100% acoperire verde).**

### Slide 11: Demonstrația Live (3 minute)
* Rularea comenzilor live din consolă și prezentarea interfeței web.

### Slide 12: Concluzii și Perspective
* O metodă științifică inovatoare pregătită pentru misiunile actuale (TESS) și viitoare (ESA PLATO, NASA Roman).

---

## 2. Scenariul Demonstrației Live (3 Minute fără Emoții)

În timpul prezentării, deschide un terminal PowerShell în folderul proiectului:

### Pasul 1: Pornirea sistemului integrat (10 secunde)
```powershell
.\scripts\start-dev.ps1
```
*Arată comisiei:* Serviciile pornesc automat pe porturile `8088` (Flutter), `8080` (Rust) și `8001` (Python).

### Pasul 2: Rularea Walkthrough-ului Științific în Terminal (30 secunde)
```powershell
.\.venv\Scripts\python.exe notebooks/pipeline_walkthrough.py
```
*Ce spui:* „Iată descărcarea live din arhiva NASA MAST pentru TOI-700, calculul scalogramei CWT, consensul hibrid AI+BLS și bateria Robovetter generând raportul de validare.”

### Pasul 3: Deschiderea Raportului HTML (20 secunde)
Deschide în browser fișierul `artifacts/walkthrough_tic_150428135.html` sau `artifacts/toi-700-report.html`.
*Ce spui:* „Raportul nu conține o simplă probabilitate oarbă, ci prezintă barele de eroare $1\sigma$, descompunerea AI vs BLS, armonicele și validarea astrofizică Gaia.”

### Pasul 4: Inspecția în Workbench-ul Flutter (60 secunde)
Deschide `http://localhost:8088`:
1. Selectează **TOI 700** pe harta cerească.
2. Apasă **„Run transit analysis”**.
3. Arată comutarea între curba brută, plierea pe fază orbitală și cardul de **Vetting Astrofizic** cu etichete verzi PASS.

---

### Pasul 5: Rularea Scannerului Batch (Opțional, dacă comisia vrea să vadă scalabilitatea)
```powershell
.\.venv\Scripts\python.exe -m astromap.cli scan --tic-ids "150428135,261136679,307210830" --output-csv artifacts/catalog.csv
```
*Ce spui:* „Sistemul scanează zeci de stele în paralel și produce automat catalogul de noi candidați.”

---

## 3. Top 10 Întrebări Dificile ale Comisiei și Răspunsurile Exacte

### Î1: „De ce ați folosit Continuous Wavelet Transform (CWT) și nu clasica Transformata Fourier Rapidă (FFT)?”
> **Răspuns:**  
> *„Transformata Fourier presupune că semnalul este staționar și pierde complet informația despre **când** a avut loc evenimentul în timp — știm frecvența, dar nu știm momentul. În schimb, CWT cu undișoară Ricker oferă o rezoluție optimă simultan în timp și în frecvență. Ea permite izolarea precisă a momentului de ingress și egress al tranzitului și distinge clar forma rotunjită de «U» a unei planete de forma ascuțită de «V» a unei binare cu eclipse.”*

### Î2: „Cum ați garantat că rețeaua nu a memorat datele (data leakage)?”
> **Răspuns:**  
> *„Am implementat un split stratificat strict grupat pe identificatorul stelei (`tic_id`). Nicio observație sau sector al unei stele din setul de test nu a apărut vreodată în setul de antrenare. Împărțirea a fost blocată determinist în `data/manifest.json`, garantând o evaluare fără scurgeri informaționale.”*

### Î3: „Ce este o pierdere Physics-Informed (PINN) și cum ajută ea rețeaua?”
> **Răspuns:**  
> *„O rețea clasică minimizează doar eroarea statistică (Binary Cross-Entropy). Ea poate învăța corelații false din zgomotul camerei. Funcția noastră de pierdere adaugă termenul $\lambda_{\text{phys}} \cdot \mathcal{L}_{\text{Mandel-Agol}}$, care penalizează tranzitele ce încalcă legile lui Kepler și ecuațiile optice de întunecare spre margine date de masa și raza stelei din catalogul Gaia. Studiul nostru de ablație demonstrează că acest termen reduce rata alarmelor false cu 62%.”*

### Î4: „De ce mai avem nevoie de algoritmul clasic BLS dacă rețeaua neuronală este atât de avansată?”
> **Răspuns:**  
> *„În știința riguroasă, AI-ul nu trebuie să înlocuiască metodele analitice verificate, ci să colaboreze cu ele. Algoritmul BLS este imbatabil în scanarea unei grile fine de frecvențe globale pe un interval de 27 de zile. Modelul nostru hibrid combină sensibilitatea morfologică a AI-ului cu rezoluția armonică a BLS-ului, verificând dacă perioada identificată este o armonică reală ($1:1, 1:2, 2:1$).”*

### Î5: „Cum demonstrează proiectul că rezolvă problema TTV (Transit Timing Variations)?”
> **Răspuns:**  
> *„Am construit un benchmark controlat (RQ3) în care am injectat decalaje temporale progresive de la 0 la 240 de minute. Datele din `artifacts/ttv_benchmark.json` arată că puterea BLS scade de la 0.258 la 0.081 (o scădere de 68%), devenind incapabil să detecteze planeta. În schimb, arhitectura noastră bazată pe atenție temporală își păstrează Recall-ul de 1.00 (100%) și o probabilitate ridicată, demonstrând că nu depinde de rigiditatea ceasornicului clasic.”*

### Î6: „Cum eliminați alarmele false provocate de stelele binare cu eclipse (Eclipsing Binaries)?”
> **Răspuns:**  
> *„Prin bateria automată de vetting (Robovetter). Primul test este Paritatea Par-Impar (Odd-Even Test). Dacă scăderea de lumină la tranzitele pare diferă cu mai mult de $3.5\sigma$ față de tranzitele impare, avem două stele de mărimi diferite care se eclipsează reciproc, iar ținta este clasificată automat drept `FALSE_POSITIVE_EB`.”*

### Î7: „Cum eliminați alarmele false provocate de companionii luminoși?”
> **Răspuns:**  
> *„Prin scanarea eclipsei secundare la anti-fază ($\phi \approx 0.5$). O exoplanetă este un corp rece și întunecat; când trece prin spatele stelei, lumina totală nu trebuie să scadă vizibil. Dacă detectăm o scădere mai mare de $2.5\sigma$, semnalul provine de la o a doua stea luminoasă din sistem, clasificată drept `FALSE_POSITIVE_SECONDARY`.”*

### Î8: „De ce ați inclus un API Gateway în Rust pe lângă codul de Python?”
> **Răspuns:**  
> *„Python este excepțional pentru matematică tensorială și PyTorch, dar are limitări de concurență (Global Interpreter Lock - GIL). Gateway-ul Axum în Rust asigură procesarea asincronă la scară a interogărilor de la sute de utilizatori, gestionează cache-ul de catalog cu zero alocări inutile de memorie și garantează stabilitatea serviciului chiar dacă un nod de inferență are o latență ridicată.”*

### Î9: „Cum ați calculat barele de eroare ($\pm \sigma$) pe parametrii fizici?”
> **Răspuns:**  
> *„Folosind o metodă de perturbare fotometrică Monte Carlo: calculăm zgomotul intrinsec al stelei (MAD - Median Absolute Deviation), injectăm perturbări controlate în flux și rulăm inferențe multiple pentru a obține dispersia standard $1\sigma$ pe perioadă, adâncime, durată și probabilitate.”*

### Î10: „Care este aplicabilitatea practică imediată a acestui software?”
> **Răspuns:**  
> *„Poate fi aplicat direct pe sectoarele 1–70 ale misiunii TESS și pe datele de arhivă Kepler pentru a identifica candidați TOI neglijați din cauza TTV-ului. Mai mult, arhitectura modulară este direct transferabilă către viitoarea misiune spațială PLATO a Agenției Spațiale Europene (ESA), programată pentru lansare în 2026.”*

---

## 4. Fraza de Încheiere (Concluzia de Aur)

> *„Domnule Președinte, Stimați Membri ai Comisiei,*  
> *AstroMap-AI demonstrează că fuziunea dintre fizica analitică diferențiabilă și inteligența artificială modernă nu este doar fezabilă, ci esențială pentru viitorul astronomiei observaționale. Am transformat o provocare teoretică într-un instrument software complet, verificat și gata de utilizare în cercetarea spațială internațională. Vă mulțumesc și vă stau la dispoziție pentru întrebări.”*
