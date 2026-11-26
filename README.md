# Wine Quality Classification — Tech Challenge Pós Tech

Projeto de classificação binária da qualidade de vinhos a partir de medições físico-químicas. O objetivo é investigar padrões nos dados e desenvolver um modelo experimental para apoiar a priorização de amostras para avaliação sensorial especializada.

> **Uso pretendido:** apoio à triagem. O modelo não certifica a qualidade do vinho, não substitui a avaliação sensorial e não demonstra relações de causa e efeito.

## Problema de negócio

A avaliação de qualidade envolve análise sensorial especializada. Este projeto investiga se medições físico-químicas podem oferecer sinais adicionais para organizar a triagem de amostras e ajudar a equipe a decidir quais vinhos merecem avaliação mais aprofundada.

A variável original `quality` foi convertida em uma classe binária:

- `1 — Alta qualidade`: `quality >= 7`.
- `0 — Baixa/média qualidade`: `quality < 7`.

## Base de dados

O arquivo `data/raw/WineQT.csv` está incluído neste repositório. Ele contém 1.143 registros, 11 variáveis físico-químicas, a nota sensorial `quality` e um identificador `Id`, excluído da modelagem.

**Fonte imediata do arquivo:** [Wine Quality Dataset — M Yasser H (Kaggle)](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset). A base está relacionada ao conjunto Wine Quality disponibilizado pelo UCI Machine Learning Repository. Para preservar a atribuição aos autores originais, este projeto também referencia a publicação de Cortez et al. (2009) e a licença CC BY 4.0 indicada pelo UCI. Consulte [`DATASET_ATTRIBUTION.md`](DATASET_ATTRIBUTION.md) para a referência completa, a licença da fonte original e as transformações realizadas neste projeto.

A análise exploratória identificou:

- Nenhum valor ausente nas colunas da base.
- 984 registros na classe baixa/média e 159 na classe alta (86,09% e 13,91%), antes da deduplicação.
- 114 perfis físico-químicos repetidos, totalizando 125 linhas excedentes; nenhum perfil repetido apresentou notas conflitantes.
- Após manter uma ocorrência por perfil, a base de modelagem ficou com 1.018 registros. A deduplicação é uma premissa metodológica conservadora: o arquivo não permite provar se os registros repetidos são duplicações de cadastro ou amostras independentes com medições coincidentes.

## Principais achados da EDA

As associações de Spearman mais destacadas com a nota original `quality` foram:

| Variável | Correlação de Spearman |
|---|---:|
| Teor alcoólico (`alcohol`) | +0,50 |
| Acidez volátil (`volatile acidity`) | -0,40 |
| Sulfatos (`sulphates`) | +0,39 |
| Ácido cítrico (`citric acid`) | +0,22 |

Essas relações descrevem associações na amostra e não demonstram causalidade. As distribuições das classes também se sobrepõem; nenhuma medição isolada determina a nota sensorial.

## Modelagem e avaliação

Foram comparadas cinco famílias de algoritmos por Nested Cross-Validation, utilizando Average Precision (AP), métrica de síntese da curva Precision-Recall, como critério principal. A configuração candidata com maior média nos folds externos foi o Random Forest com 13 variáveis (11 originais e 2 derivadas), com AP média de 0,5804 (desvio-padrão de 0,0426).

As duas variáveis derivadas foram:

- `free_sulfur_ratio = free sulfur dioxide / total sulfur dioxide`
- `volatile_to_fixed_acidity = volatile acidity / fixed acidity`

Após o ajuste final apenas no conjunto de treino, a avaliação no holdout (204 observações, incluindo 27 positivas) apresentou Average Precision de 0,5504 e ROC-AUC de 0,8818.

### Cenários de threshold no conjunto de teste

| Cenário | Threshold | Precision | Recall | F1-Score | VP | FP | FN | VN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Padrão | 0,50 | 61,54% | 29,63% | 40,00% | 8 | 5 | 19 | 172 |
| Intermediário | 0,60 | 71,43% | 18,52% | 29,41% | 5 | 2 | 22 | 175 |
| Conservador | 0,70 | 100,00% | 3,70% | 7,14% | 1 | 0 | 26 | 177 |

Nenhum threshold foi declarado universalmente ideal. O cenário deve ser escolhido com a área de negócio, considerando os custos de falsos positivos e falsos negativos e a capacidade de avaliação sensorial. A Precision de 100% no cenário 0,70 corresponde a uma única previsão positiva e não deve ser interpretada como garantia de desempenho.

## Interpretabilidade

A Permutation Feature Importance individual destacou `alcohol`, `sulphates` e `free_sulfur_ratio`. Na análise por grupos, destacaram-se os grupos “Álcool e densidade” e “Acidez”. Essas análises medem a dependência preditiva do modelo nesta amostra; não indicam causalidade. A amostra de teste contém poucas observações positivas e os resultados de importância devem ser tratados como exploratórios.

## Estrutura do repositório

```text
wine-quality-classification/
├── data/
│   ├── raw/
│   │   └── WineQT.csv
│   └── examples/
│       ├── novos_vinhos_exemplo.csv
│       └── threshold_demo_from_holdout.csv
├── notebooks/
│   └── 01_wine_quality_classification_final.ipynb
├── src/
│   ├── __init__.py
│   ├── features.py
│   └── predict.py
├── models/
│   └── wine_quality_model.pkl
├── tests/
│   ├── test_features.py
│   └── test_predict.py
├── results/
│   └── predictions/
│       ├── threshold_demo_0_50.csv
│       ├── threshold_demo_0_60.csv
│       ├── threshold_demo_0_70.csv
│       └── threshold_demo_comparison.csv
├── requirements.txt
├── LICENSE
├── DATASET_ATTRIBUTION.md
├── .gitignore
├── .gitattributes
└── README.md
```

## Como executar no Windows PowerShell

### 1. Criar e ativar o ambiente virtual

Na raiz do projeto, confira a versão do Python:

```powershell
python --version
```

O ambiente utilizado no desenvolvimento foi Python `3.15.0`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear a ativação, execute na sessão atual:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 2. Instalar dependências

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

### 3. Executar o notebook

Abra `notebooks/01_wine_quality_classification_final.ipynb` no VS Code, selecione o ambiente `.venv` como kernel e execute as células em ordem. A seção 8 recria o artefato `models/wine_quality_model.pkl`; a seção 9 confere o artefato e gera arquivos demonstrativos para comparar os thresholds.

O arquivo Pickle já está incluído no repositório para permitir executar predições sem treinar o modelo novamente. Para reproduzir a análise e recriar o artefato, execute o notebook completo.

### 4. Executar os testes automatizados

Com o ambiente virtual ativado e na raiz do projeto:

```powershell
python -m pytest -q
```

Os testes verificam a criação das duas variáveis derivadas, a validação de entradas e a alteração da classe prevista quando o threshold muda, mantendo o score constante.

### 5. Fazer predições com um novo CSV

A partir da raiz do projeto, com o ambiente virtual ativado:

```powershell
python -m src.predict --input data/examples/novos_vinhos_exemplo.csv --threshold 0.50
```

Os thresholds disponíveis são `0.50`, `0.60` e `0.70`. Para definir outro arquivo de saída:

```powershell
python -m src.predict --input data/examples/novos_vinhos_exemplo.csv --threshold 0.60 --output results/predictions/lote_001.csv
```

### Formato de entrada

O CSV deve conter as 11 colunas físico-químicas com os mesmos nomes usados no treinamento:

`fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`.

A coluna `sample_id` é opcional e serve para identificar a amostra. Não inclua `quality` ou `target`, pois essas informações não estarão disponíveis quando o modelo receber dados futuros.

### Formato de saída

O arquivo de saída mantém as medições originais e acrescenta:

- `score_modelo_alta_qualidade`: score retornado por `predict_proba`; não foi demonstrado que seja uma probabilidade calibrada.
- `threshold_aplicado`: limite utilizado na execução.
- `target_previsto`: classe prevista (0 ou 1).
- `classificacao_prevista`: rótulo textual da classe.

## Licença

O código-fonte original deste projeto é disponibilizado sob a licença MIT, conforme o arquivo [`LICENSE`](LICENSE). Essa licença se aplica ao código deste repositório e não substitui nem altera os termos de materiais de terceiros.

O dataset não é licenciado pelo arquivo MIT. Sua fonte imediata e a atribuição aos autores do conjunto original, incluindo a licença CC BY 4.0 indicada pelo UCI, estão documentadas em [`DATASET_ATTRIBUTION.md`](DATASET_ATTRIBUTION.md).

O modelo serializado (`models/wine_quality_model.pkl`) é um artefato experimental treinado com esse conjunto de dados. A licença MIT do código não deve ser interpretada como uma declaração de licença independente para dados de terceiros ou como garantia de direitos adicionais sobre o artefato treinado.

## Ambiente e dependências

O projeto foi desenvolvido e validado no seguinte ambiente:

- Python: `3.15.0`
- NumPy: `2.5.5`
- Pandas: `3.1.1`
- SciPy: `1.18.1`
- Scikit-learn: `1.9.1`
- Matplotlib: `3.11.2`
- Seaborn: `0.13.2`
- Statsmodels: `0.15.0`
- Pytest: `9.2.0`

As dependências e versões estão registradas em `requirements.txt`. O carregamento de modelos serializados entre versões diferentes de Python e Scikit-learn não é garantido. O Pickle deve ser carregado somente quando foi criado pelo próprio projeto ou por uma fonte confiável, pois sua desserialização pode executar código.

## Limitações e uso responsável

- O conjunto de teste contém apenas 27 observações positivas; pequenas mudanças na classificação alteram sensivelmente Precision e Recall.
- A validação cruzada e o holdout são avaliações internas e não substituem testes em novos lotes, safras, produtores ou equipamentos.
- Os scores de `predict_proba` não foram avaliados quanto à calibração.
- A importância das variáveis é preditiva, não causal.
- A seleção do threshold depende de custos e capacidade operacional ainda não quantificados.
- O modelo deve apoiar a triagem e a priorização de análises; a avaliação sensorial especializada continua necessária.

Projeto acadêmico desenvolvido para o Tech Challenge da Pós Tech.
