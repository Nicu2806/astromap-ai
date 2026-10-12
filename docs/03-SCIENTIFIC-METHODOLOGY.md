# Metodologia științifică

## 1. Principiul de raportare

Metoda software și rezultatul științific sunt lucruri diferite. Codul implementează un pipeline și o arhitectură neurală, dar proiectul nu deține încă un checkpoint validat pe un dataset real etichetat. În prezent:

- datele analizate live sunt reale TESS/SPOC;
- metoda implicită este BLS;
- fixture-urile și smoke training-ul sunt sintetice;
- orice rezultat AI trebuie numit nevalidat.

## 2. Sursa datelor

### NASA MAST

`mast.py` caută produse cu:

- target: `TIC {id}`;
- mission: `TESS`;
- author: `SPOC`;
- sector opțional.

Inventarul expune un produs preferat per sector. Dacă există mai multe expuneri, produsul cu expunerea cea mai apropiată de 120 s este ales. Download-ul utilizează `quality_bitmask="default"`, apoi normalizează curba prin Lightkurve.

### Proveniență minimă

Fiecare rezultat live include:

- TIC ID;
- `mission = TESS/SPOC`;
- sector;
- numărul de cadențe;
- URI MAST `mast:TESS/product/...fits`;
- versiunea metodei/checkpoint-ului.

## 3. Reprezentarea curbei

`LightCurve` conține patru vectori de aceeași lungime:

$$
(t_i, f_i, \sigma_i, q_i), \quad i=1,\ldots,N
$$

unde $t$ este timpul, $f$ fluxul, $\sigma$ eroarea fluxului și $q$ quality flag. Constructorul respinge vectori goi sau cu lungimi diferite.

## 4. Preprocesarea implementată

Funcția `preprocess()` execută:

1. eliminarea valorilor nefinite;
2. păstrarea cadențelor cu `quality == 0`;
3. respingerea seriilor cu mai puțin de 32 puncte valide;
4. sortarea temporală și eliminarea timpilor duplicați;
5. detrending Savitzky-Golay, ordin 2, fereastră impară de maximum 101;
6. împărțirea fluxului la trend și normalizarea medianei la 1;
7. resampling liniar la 512 puncte;
8. construirea unei măști pentru gap-uri mai mari de 2,5 cadențe mediane;
9. înlocuirea punctelor mascate cu flux neutru 1;
10. calculul scalogramei Ricker.

Fluxul normalizat este aproximativ:

$$
f_{norm}(t) = \frac{f(t)}{SG(f(t))} \cdot
\frac{1}{\operatorname{median}(f/SG(f))}.
$$

### Limitări

- fereastra Savitzky-Golay nu este aleasă din durata tranzitului;
- gap-urile sunt interpolate înainte de mascarea intrării;
- o singură grilă de 512 puncte reduce informația din curbe lungi;
- sectorul este procesat individual, fără stitching multi-sector;
- flux error nu intră încă în model;
- nu există centroid features sau quality channels detaliate.

## 5. Scalograma

`ricker_scalogram()` folosește 32 scale geometrice între 1,5 și 48. Pentru fiecare scală $s$ este construit un wavelet Ricker discret:

$$
\psi_s(x) = \left(1 - \left(\frac{x}{s}\right)^2\right)
\exp\left(-\frac{x^2}{2s^2}\right).
$$

Wavelet-ul este centrat și normalizat L2, apoi convoluat cu $f_{norm}-1$. Scalograma finală este împărțită la deviația standard globală.

Aceasta este o transformare CWT-like implementată local, nu apelul unei biblioteci CWT complete.

## 6. Baseline-ul BLS

În lipsa unui checkpoint, `_bls_estimate()` folosește `astropy.timeseries.BoxLeastSquares`.

Grila implementată:

- perioade: 1.000 valori între 0,5 zile și $\min(13, T_{baseline}/2)$;
- durate: 0,05, 0,1, 0,15, 0,2 și 0,3 zile;
- selecție: puterea BLS maximă.

Zgomotul punctual robust este aproximat prin:

$$
\sigma_{point} \approx \frac{\operatorname{median}(|\Delta f|)}{0.9539}.
$$

Scorul de screening este o mapare logistică a SNR-ului aproximat:

$$
score = \frac{1}{1 + \exp(-(SNR - 7)/2)}.
$$

Acest scor nu este o probabilitate calibrată. Pragul de 0,5 produce doar etichetele operaționale `candidate` și `unlikely`.

### Limitări BLS

- un sector poate conține prea puține tranzite pentru perioada reală;
- grila nu include perioade sub 0,5 sau peste limita sectorului;
- sistematicile și binarele eclipsante pot produce maxime puternice;
- scorul logistic nu a fost calibrat pe un catalog etichetat;
- rezultatul trebuie comparat cu profilul phase-folded și metadata stelei.

## 7. Modelul `PinnTransNet`

Modelul actual este dual-domain, fără TCN și fără attention.

### Ramura temporală

Intrare: două canale, fluxul centrat și masca.

```text
Conv1D 2->24, k=9
GELU
MaxPool1D 4
Conv1D 24->48, k=7
GELU
AdaptiveAvgPool1D 16
```

Rezultat: $48 \times 16 = 768$ caracteristici.

### Ramura wavelet

Intrare: scalograma $1 \times 32 \times 512$.

```text
Conv2D 1->16, k=5
GELU
MaxPool2D (2,4)
Conv2D 16->24, k=3
GELU
AdaptiveAvgPool2D (4,8)
```

Rezultat: $24 \times 4 \times 8 = 768$ caracteristici.

### Fuziune și heads

- concatenare la 1.536 caracteristici;
- `Linear(1536, 128) + GELU`;
- classifier `Linear(128, 1)` produce logits;
- regressor `Linear(128, 3) + softplus` produce depth, duration și period pozitive.

## 8. Loss-ul fizic

`physics_loss()` primește parametrii estimați și raza/masa stelei. Din legea a treia a lui Kepler:

$$
a^3 = \frac{G M_\star P^2}{4\pi^2}.
$$

Pentru un tranzit central și orbită circulară se calculează o durată maximă conservatoare. Loss-ul penalizează:

$$
\mathcal{L}_{phys} = \mathbb{E}\left[
\operatorname{ReLU}(T_{dur}-T_{max})^2 +
\operatorname{ReLU}(T_{dur}-P)^2
\right].
$$

Nu este implementarea completă Mandel-Agol. Nu estimează excentricitatea, impact parameter sau limb darkening.

## 9. Training-ul existent

`training.py` construiește un batch sintetic echilibrat, rulează câteva epoci și salvează un `state_dict`. Este destinat validării mecanice a:

- formelor tensorilor;
- forward/backward;
- optimizerului AdamW;
- gradient clipping;
- serializării checkpoint-ului.

Nu trebuie folosit în teză drept antrenare reală și nu trebuie promovat în deployment.

## 10. Protocolul necesar pentru modelul real

1. importă cataloagele NASA Exoplanet Archive cu versiune;
2. construiește clase confirmed/candidate/false positive/non-transit;
3. face cross-match TIC/Gaia cu proveniență;
4. îngheață manifestul și deduplică țintele;
5. împarte train/validation/test prin TIC ID;
6. calculează statisticile exclusiv pe train;
7. antrenează baseline 1D și modelul dual-domain;
8. rulează ablații pentru CWT și physics loss;
9. selectează checkpoint-ul numai pe validation;
10. deschide test set-ul o singură dată pentru evaluarea finală.

Metricile minime sunt PR-AUC, ROC-AUC, FPR la 95% recall, Brier score, calibrare, MAE pentru parametri și intervale bootstrap la nivel de stea.

## 11. Reguli de interpretare

- `bls_screening_score`: valoare euristică derivată din BLS/SNR;
- `unvalidated_ai_probability`: sigmoid model, încă necalibrat și nevalidat;
- `candidate`: etichetă de screening internă;
- niciunul dintre termenii de mai sus nu înseamnă „confirmed planet”.

Orice raport sau UI trebuie să păstreze disclaimer-ul și versiunea metodei.