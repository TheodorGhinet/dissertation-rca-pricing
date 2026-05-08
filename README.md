# De la modele liniare la rețele neuronale interpretabile
### Un studiu comparativ pentru tarifarea riscului în asigurările auto (RCA)

Cod sursă aferent lucrării de disertație, Master Statistică Aplicată și Data Science,  
Universitatea din București, Facultatea de Matematică și Informatică, 2025.

---

## Descriere

Lucrarea evaluează comparativ **9 arhitecturi de modele predictive** pentru tarifarea
riscului în asigurările auto obligatorii (RCA) pe un portofoliu românesc de 698.014 polițe,
utilizând decompunerea actuarială standard frecvență–severitate.

| Model | Tip |
|---|---|
| GLM (Poisson + Gamma) | Baseline actuarial |
| LightGBM | Gradient Boosting |
| XGBoost | Gradient Boosting |
| CANN | Hibrid GLM + rețea neuronală |
| LocalGLMnet Binned | Interpretabil — coeficienți locali |
| LocalGLMnet Mixt | Interpretabil — coeficienți locali |
| PureFTT | Transformer tabular pur |
| CAFTT | Hibrid GLM + FT-Transformer |
| LocalGLM-FTT | Interpretabil — coeficienți locali + FT-Transformer |

---

## Structura repository-ului

```
dissertation-rca-pricing/
│
├── rca_pricing_models.ipynb     # Notebook principal: preprocesare, modele, evaluare
├── requirements.txt             # Versiuni pachete Python
└── README.md
```

> **Notă privind datele:** Din motive de confidențialitate, datele brute ale portofoliului
> RCA nu sunt incluse în acest repository. Notebook-ul este structurat astfel încât
> secțiunile de modelare și evaluare pot fi inspectate independent de rularea efectivă.

---

## Conținutul notebook-ului

Notebook-ul `rca_pricing_models.ipynb` este organizat în următoarele secțiuni:

1. **Configurare și importuri** — librării, seed-uri, parametri globali
2. **Încărcare și curățare date** — filtrare valori lipsă și extreme, capping frecvență P99,5
3. **Ingineria variabilelor** — binning bazat pe arbori de decizie, one-hot encoding
4. **Split antrenament / testare** — split stratificat 80/20
5. **GLM baseline** — Poisson (frecvență) și Gamma (severitate) cu statsmodels
6. **LightGBM și XGBoost** — gradient boosting cu early stopping
7. **CANN** — Combined Actuarial Neural Network cu offset GLM
8. **LocalGLMnet** — varianta Binned și varianta Mixtă
9. **FT-Transformer** — PureFTT, CAFTT și LocalGLM-FTT
10. **Evaluare comparativă** — devianță Poisson/Gamma, A/E ratio, grafice marginale

---

## Rulare

### Google Colab (recomandat)
Deschide direct în Colab cu butonul de mai jos — nu necesită instalare locală.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheodorGhinet/dissertation-rca-pricing/blob/main/rca_pricing_models.ipynb)

### Local
```bash
git clone https://github.com/TheodorGhinet/dissertation-rca-pricing.git
cd dissertation-rca-pricing
pip install -r requirements.txt
jupyter notebook rca_pricing_models.ipynb
```

---

## Rezultate principale

| Model | Devianță Poisson (test) | Devianță Gamma (test) | A/E (test) |
|---|---|---|---|
| GLM | 0,276998 | 1,186374 | 0,9409 |
| CANN | 0,276994 | 1,187085 | 0,9948 |
| CAFTT | 0,277038 | **1,184654** | 0,9550 |
| LightGBM | 0,277084 | 1,193455 | 0,9561 |
| LocalGLMnet Mixt | 0,277302 | 1,198895 | 1,0034 |
| PureFTT | 1,788550 ❌ | 1,377546 | 5,8679 |

---

## Cerințe tehnice

- Python 3.12
- GPU recomandat pentru secțiunile de rețele neuronale (testat pe NVIDIA Tesla T4 — Google Colab)
- RAM minim 8 GB

---

## Referințe principale

- Wüthrich, M. V. (2022). The balance property in neural network modelling. *Statistical Theory and Related Fields*, 6(1), 1–9.
- Richman, R. & Wüthrich, M. V. (2023). LocalGLMnet: interpretable deep learning for tabular data. *Scandinavian Actuarial Journal*, 2023(1), 71–95.
- Brauer, A. (2024). Enhancing actuarial non-life pricing models via transformers. arXiv:2311.07597.
- Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022). Why tree-based models still outperform deep learning on tabular data. *NeurIPS 35*, 507–520.
