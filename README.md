# dissertation-rca-pricing
# De la modele liniare la rețele neuronale interpretabile
### Un studiu comparativ pentru tarifarea riscului în asigurările auto (RCA)

Cod sursă aferent lucrării de disertație, Master Statistică Aplicată și Data Science,  
Universitatea din București, Facultatea de Matematică și Informatică, 2025.

---

## Descriere

Lucrarea evaluează comparativ **9 arhitecturi de modele predictive** pentru tarifarea
riscului în asigurările auto obligatorii (RCA) pe un portofoliu românesc de 698.014 polițe,
utilizând decompunerea actuarială standard frecvență–severitate.

Arhitecturile comparate:

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
rca-tarifare/
│
├── 01_preprocesare.ipynb        # Curățarea datelor și ingineria variabilelor
├── 02_glm_baseline.ipynb        # GLM Poisson (frecvență) și Gamma (severitate)
├── 03_gradient_boosting.ipynb   # LightGBM și XGBoost
├── 04_retele_neuronale.ipynb    # CANN și LocalGLMnet (Binned + Mixt)
├── 05_transformer.ipynb         # PureFTT, CAFTT și LocalGLM-FTT
├── 06_evaluare_comparativa.ipynb# Metrici, grafice comparative, A/E ratio
│
├── requirements.txt             # Versiuni pachete Python
└── README.md
```

> **Notă privind datele:** Din motive de confidențialitate, datele brute ale portofoliului
> RCA nu sunt incluse în acest repository. Codul este funcțional pe orice portofoliu
> structurat conform schemei descrise în `01_preprocesare.ipynb`.

---

## Instalare și rulare

### 1. Clonează repository-ul
```bash
git clone https://github.com/NumeleTau/rca-tarifare.git
cd rca-tarifare
```

### 2. Instalează dependențele
```bash
pip install -r requirements.txt
```

### 3. Rulează notebook-urile în ordine
```bash
jupyter notebook
```
Deschide și rulează notebook-urile în ordine numerică (01 → 06).

### Alternativ — Google Colab
Fiecare notebook poate fi rulat direct în Google Colab fără instalare locală.
Click pe badge-ul [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
din fiecare notebook.

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

Concluzie: pe un spațiu de 7 variabile explicative, toate arhitecturile funcționale
converg la performanțe predictive practic identice cu GLM-ul.

---

## Cerințe tehnice

- Python 3.12+
- GPU recomandat pentru notebook-urile 04 și 05 (testat pe NVIDIA Tesla T4 — Google Colab)
- RAM minim 8 GB

---

## Referințe

Lucrarea se bazează pe, printre altele:

- Wüthrich, M. V. (2022). The balance property in neural network modelling. *Statistical Theory and Related Fields*, 6(1), 1–9.
- Richman, R. & Wüthrich, M. V. (2023). LocalGLMnet: interpretable deep learning for tabular data. *Scandinavian Actuarial Journal*, 2023(1), 71–95.
- Brauer, A. (2024). Enhancing actuarial non-life pricing models via transformers. arXiv:2311.07597.
- Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022). Why tree-based models still outperform deep learning on tabular data. *NeurIPS 35*, 507–520.

Lista completă a referințelor se regăsește în lucrarea de disertație.
