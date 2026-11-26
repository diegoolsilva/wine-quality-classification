# Classificação Binária de Qualidade de Vinhos (`Wine Quality`)

Projeto end-to-end de **Machine Learning (Classificação Binária)** aplicado ao dataset `WineQT.csv`, estruturado com foco em boas práticas metodológicas (prevenção de *Data Leakage*, validação cruzada estratificada, *Feature Engineering*, otimização de hiperparâmetros e ajuste de limiar orientado a regra de negócio).

## 1. Problema e Premissa de Negócio (Cenário A)
A variável original `quality` foi convertida em um alvo binário:
- **`1` (Alta Qualidade):** `quality >= 7`
- **`0` (Baixa/Média Qualidade):** `quality < 7`

O conjunto apresenta **desbalanceamento estrutural** (cerca de **13,5%** de amostras positivas). Sob o **Cenário A** (selo e precificação *premium*), classificar indevidamente um vinho comum como Alta Qualidade (**Falso Positivo — FP**) traz maior risco reputacional e comercial do que deixar de aprovar um candidato válido (**Falso Negativo — FN**). Por isso, o projeto priorizou **PR-AUC** e **Precision**, sem ignorar *Recall*, *F1-Score* e *ROC-AUC*.

## 2. Metodologia e Pipeline do Projeto
1. **Setup e Reprodutibilidade:** semente fixa (`RANDOM_STATE = 2024`) e estrutura organizada em Jupyter Notebook (`notebooks/01_wine_quality_classification.ipynb`).
2. **Análise Exploratória (EDA):** identificação de distribuições assimétricas, *outliers* e **125 duplicatas exatas** em um total inicial de **1.143 registros**.
3. **Pré-processamento e *Feature Engineering*:**
   - Remoção de duplicatas antes da divisão dos dados (**1.018 registros únicos**).
   - *Stratified Train-Test Split* (`80/20` -> **814 treino** / **204 teste**).
   - Diagnóstico de multicolinearidade via **VIF** exclusivamente no treino.
   - Criação de 2 variáveis de domínio (`free_sulfur_ratio` e `volatile_to_fixed_acidity`), totalizando **13 atributos** (`X_train_fe` / `X_test_fe`).
   - Uso de `Pipeline` do `scikit-learn` para evitar vazamento de dados no escalonamento e no ajuste dos modelos.
4. **Modelagem e Otimização (`GridSearchCV`):** comparação de *baselines* (`DummyClassifier` e `Logistic Regression`) e otimização unificada com *Stratified 5-Fold CV* (`average_precision` / *PR-AUC*) em **Regressão Logística**, **KNN**, **SVM (RBF)**, **Random Forest** e **Gradient Boosting** nos cenários de **11 vs. 13 variáveis**.
5. **Avaliação, Threshold Tuning e Teste Final:** diagnóstico de *overfitting* (*Train vs. CV PR-AUC*), análise de curvas de limiar em previsões *out-of-fold* e validação final no conjunto de teste (`X_test_fe`).

## 3. Principais Resultados
- **Impacto das 13 variáveis:** os modelos avaliados com o conjunto expandido (`13 Features (FE)`) mantiveram ou superaram o desempenho frente às 11 variáveis originais na validação cruzada, com o **Random Forest** (`n_estimators=300`, `max_depth=6`, `min_samples_leaf=1`) alcançando **`0,6091` de CV PR-AUC** e o **SVM (RBF)** figurando logo em seguida na validação cruzada.
- **Escolha de limiar (`threshold = 0.60` no SVM RBF):** no conjunto de teste (`X_test_fe`, 204 amostras), elevar o limiar de `0.50` para `0.60` reduziu os **Falsos Positivos de 5 para 2** e aumentou a **Precision de `0,6875` para `0,8462`**, mantendo boa generalização em dados inéditos.

## 4. Estrutura do Repositório
```text
├── notebooks/
│   └── 01_wine_quality_classification.ipynb
├── WineQT.csv
└── README.md