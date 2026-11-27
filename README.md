# Wine Quality Classification — Tech Challenge Pós Tech

Projeto de classificação binária da qualidade de vinhos a partir de medições físico-químicas. O objetivo é investigar padrões nos dados e desenvolver um modelo experimental para **apoiar a triagem de amostras que mereçam avaliação sensorial especializada**.

> **Uso pretendido:** apoio à triagem, não certificação. O modelo procura sinalizar vinhos potencialmente pertencentes à classe de alta qualidade; a decisão final continua dependendo de avaliação sensorial especializada.

## Problema de negócio

A avaliação de qualidade envolve análise sensorial especializada. Este projeto investiga se medições físico-químicas podem ajudar a priorizar amostras para uma avaliação mais aprofundada.

A variável original `quality` foi convertida em uma classe binária:

- `1 — Alta qualidade`: nota `quality >= 7`.
- `0 — Baixa/média qualidade`: nota `quality < 7`.

O foco operacional escolhido é a **triagem**. Por isso, o threshold foi selecionado com a métrica F2, que dá mais peso ao Recall. Essa escolha tende a sinalizar mais amostras, incluindo falsos positivos, para reduzir a quantidade de vinhos de alta qualidade que passam despercebidos.

## Base de dados

O arquivo utilizado é `data/raw/WineQT.csv`, obtido da versão disponibilizada por M Yasser H no Kaggle. Ele contém 1.143 registros, 11 variáveis físico-químicas, a nota sensorial `quality` e o identificador `Id`. O identificador foi excluído da modelagem.

O arquivo corresponde a um recorte associado a **vinhos tintos**. Os resultados não devem ser generalizados para vinhos brancos sem uma avaliação específica.

### Principais achados da análise exploratória

- Não foram identificados valores ausentes nas colunas da base.
- Antes da deduplicação, havia 984 registros na classe baixa/média e 159 na classe alta (86,09% e 13,91%).
- Foram identificados 114 perfis físico-químicos repetidos, totalizando 125 linhas excedentes; nenhum perfil repetido apresentou notas conflitantes.
- Após manter uma ocorrência por perfil, a base utilizada na modelagem ficou com 1.018 registros.
- A deduplicação é uma premissa metodológica conservadora: o arquivo não permite provar se os registros repetidos são duplicações de cadastro ou amostras independentes com medições coincidentes.

As associações de Spearman mais destacadas com a nota original `quality` foram:

| Variável | Correlação de Spearman |
|---|---:|
| Teor alcoólico (`alcohol`) | +0,50 |
| Acidez volátil (`volatile acidity`) | -0,40 |
| Sulfatos (`sulphates`) | +0,39 |
| Ácido cítrico (`citric acid`) | +0,22 |

Essas correlações descrevem associações observadas na amostra; não demonstram relações de causa e efeito.

## Preparação e engenharia de atributos

Foram criadas duas variáveis derivadas:

- `free_sulfur_ratio = free sulfur dioxide / total sulfur dioxide`
- `volatile_to_fixed_acidity = volatile acidity / fixed acidity`

A base foi dividida de forma estratificada em treino e teste. Após a deduplicação, foram utilizadas 814 observações no treino e 204 no teste. A classe positiva no teste contém 27 observações.

Os valores extremos foram quantificados pelo critério de 1,5 IQR. Os registros foram mantidos porque valores fora dos limites estatísticos não são, por si só, evidência de erro de medição. Também foi verificada a consistência `free sulfur dioxide <= total sulfur dioxide`.

## Modelagem

Foram comparadas cinco famílias de algoritmos por validação cruzada estratificada de cinco folds, usando Average Precision (AP) como métrica principal:

| Algoritmo | AP média na comparação exploratória |
|---|---:|
| SVM RBF | **0,5948** |
| Random Forest | 0,5823 |
| Regressão Logística | 0,5552 |
| Gradient Boosting | 0,5479 |
| KNN | 0,4669 |

Depois da comparação exploratória, foi feita uma busca pequena de hiperparâmetros nos três candidatos selecionados, mantendo o Random Forest entre eles para avaliar o balanceamento das classes.

| Modelo ajustado | Melhor AP média na validação cruzada |
|---|---:|
| Random Forest | **0,6179** |
| SVM RBF | 0,6033 |
| Regressão Logística | 0,5669 |

O modelo selecionado foi o **Random Forest**, com:

```python
{
    "n_estimators": 200,
    "max_depth": 5,
    "min_samples_leaf": 3,
    "class_weight": "balanced_subsample"
}
```

A comparação das configurações do Random Forest indicou que `max_depth=5` apresentou a maior AP média de validação entre as opções avaliadas, com menor diferença entre AP de treino e validação do que a configuração de profundidade 6.

### Diagnóstico de ajuste excessivo (*overfitting*)

Na configuração selecionada, a AP média nos folds de treinamento foi 0,8979 e a AP média nos folds de validação foi 0,6179. A diferença indica sinal de ajuste excessivo. O resultado é apresentado como diagnóstico, não como prova isolada de que o modelo falhará em dados futuros.

## Escolha do threshold para triagem

O threshold foi escolhido **somente com previsões out-of-fold (OOF) da partição de treino**, maximizando F2. Como os hiperparâmetros também foram selecionados por validação cruzada nessa mesma partição, as métricas OOF são usadas como ferramenta interna de seleção do threshold, não como uma validação independente adicional.

O threshold recomendado nesta execução foi **0,39**:

| Métrica OOF | Resultado |
|---|---:|
| Precision | 42,72% |
| Recall | 82,73% |
| F2-Score | 0,6968 |
| Amostras sinalizadas | 213 |

### Avaliação no conjunto de teste

O modelo final, ajustado no conjunto de treino, foi avaliado no holdout de 204 observações. A Average Precision foi **0,6122** e a ROC-AUC foi **0,8845**.

No threshold recomendado de 0,39:

| Métrica | Resultado |
|---|---:|
| Precision | 34,43% |
| Recall | **77,78%** |
| F1-Score | 47,73% |
| F2-Score | 62,13% |
| Amostras sinalizadas | 61 |
| Verdadeiros positivos | 21 |
| Falsos positivos | 40 |
| Falsos negativos | 6 |
| Verdadeiros negativos | 137 |

Nesse teste, o modelo sinalizou 61 das 204 amostras e identificou 21 dos 27 vinhos da classe positiva. Em contrapartida, 40 amostras sinalizadas não pertenciam à classe de alta qualidade. Esse resultado é compatível com uma estratégia de triagem que prioriza Recall, mas sua viabilidade depende da capacidade da equipe para realizar avaliações sensoriais adicionais.

Os thresholds 0,50, 0,60 e 0,70 são apresentados como análise de sensibilidade, não como alternativas usadas para escolher o resultado depois de consultar o teste:

| Threshold | Precision | Recall | F1-Score | Amostras sinalizadas |
|---:|---:|---:|---:|---:|
| **0,39 — recomendado (F2)** | 34,43% | **77,78%** | 47,73% | 61 |
| 0,50 | 36,00% | 66,67% | 46,75% | 50 |
| 0,60 | 44,44% | 59,26% | **50,79%** | 36 |
| 0,70 | 61,11% | 40,74% | 48,89% | 18 |

A recomendação de 0,39 prioriza a identificação de mais vinhos potencialmente bons. A empresa deve avaliar se o volume de falsos positivos é operacionalmente aceitável.

## Interpretabilidade

A Permutation Feature Importance individual destacou as seguintes variáveis:

| Variável | Queda média na AP |
|---|---:|
| `sulphates` | 0,1389 |
| `alcohol` | 0,1363 |
| `volatile acidity` | 0,0735 |
| `citric acid` | 0,0644 |
| `total sulfur dioxide` | 0,0505 |

A importância por permutação indica quanto o desempenho preditivo do modelo diminui quando a informação de uma variável é embaralhada. Ela não indica causalidade. Variáveis correlacionadas podem compartilhar informação, fazendo com que a importância individual de uma delas pareça menor.

## Possíveis implicações para a produção

Os resultados podem orientar hipóteses de monitoramento, a serem avaliadas por profissionais da produção e enologia:

- acompanhar a acidez volátil, por sua associação observada com a nota e sua contribuição para as previsões;
- monitorar teor alcoólico e sulfatos como características que ajudaram o modelo a diferenciar as classes;
- acompanhar a relação entre dióxido de enxofre livre e total como indicador complementar do perfil físico-químico.

Esses achados não justificam, isoladamente, alterar receitas ou parâmetros de produção. Podem orientar investigações e monitoramento, sempre combinados com conhecimento enológico e avaliação sensorial.

## Estrutura do repositório

```text
wine-quality-classification/
├── data/
│   ├── raw/
│   │   └── WineQT.csv
│   └── examples/
│       ├── novos_vinhos_exemplo.csv
│       └── threshold_demo_from_holdout.csv
├── models/
│   └── wine_quality_model.pkl
├── notebooks/
│   └── 01_wine_quality_classification_final.ipynb
├── results/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── src/
│   ├── __init__.py
│   ├── features.py
│   └── predict.py
├── tests/
│   ├── test_features.py
│   └── test_predict.py
├── DATASET_ATTRIBUTION.md
├── LICENSE
├── README.md
├── requirements.txt
└── .gitignore
```

## Como executar no Windows PowerShell

### 1. Criar e ativar o ambiente virtual

Na raiz do projeto:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

O ambiente usado para gerar o modelo Pickle foi Python `3.15.0`. Para reproduzir exatamente o artefato serializado, utilize também versões compatíveis das bibliotecas listadas em `requirements.txt`.

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

Abra `notebooks/01_wine_quality_classification_final.ipynb` no VS Code, selecione o ambiente `.venv` como kernel e execute as células em ordem. A seção 8 recria o arquivo `models/wine_quality_model.pkl`.

### 4. Executar os testes automatizados

Com o ambiente virtual ativado e na raiz do projeto:

```powershell
python -m pytest -q
```

### 5. Fazer predições com um novo CSV

O script usa automaticamente o threshold recomendado salvo no Pickle quando `--threshold` não é informado:

```powershell
python -m src.predict --input data/examples/novos_vinhos_exemplo.csv
```

Também é possível informar manualmente qualquer threshold entre 0 e 1:

```powershell
python -m src.predict --input data/examples/novos_vinhos_exemplo.csv --threshold 0.50
```

Para escolher o arquivo de saída:

```powershell
python -m src.predict --input data/examples/novos_vinhos_exemplo.csv --output results/predictions/lote_001.csv
```

### Formato de entrada

O CSV deve conter as 11 colunas físico-químicas com os mesmos nomes usados no treinamento:

`fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`.

A coluna `sample_id` é opcional e serve para identificar a amostra. Não inclua `quality` ou `target`, pois essas informações não estarão disponíveis ao realizar previsões futuras.

### Formato de saída

O arquivo de saída mantém as medições originais e acrescenta:

- `score_modelo_alta_qualidade`: score retornado por `predict_proba`; não foi demonstrado que seja uma probabilidade calibrada.
- `threshold_aplicado`: limite utilizado na execução.
- `target_previsto`: classe prevista (0 ou 1).
- `classificacao_prevista`: rótulo textual da classe.

## Limitações

- O conjunto de teste contém apenas 27 observações positivas; pequenas mudanças na classificação alteram sensivelmente Precision e Recall.
- O conjunto utilizado corresponde a um recorte associado a vinhos tintos; o desempenho em vinhos brancos não foi avaliado.
- A deduplicação é uma decisão metodológica, pois não é possível confirmar se todos os perfis repetidos representam a mesma amostra física.
- A diferença entre AP de treino e validação indica possível overfitting.
- A seleção de hiperparâmetros e threshold foi feita dentro da partição de treino por validação cruzada; o holdout foi reservado para a avaliação final. Após essa avaliação, o holdout também foi utilizado para a análise exploratória de importância por permutação e para selecionar exemplos demonstrativos. Essas análises posteriores não foram usadas para retreinar o modelo nem para escolher o threshold.
- Os scores de `predict_proba` não foram avaliados quanto à calibração.
- A importância das variáveis é preditiva, não causal.
- A escolha do threshold depende dos custos dos erros e da capacidade operacional da vinícola, ainda não quantificados.
- O modelo é experimental e deve apoiar a triagem; a avaliação sensorial especializada continua necessária.

## Licença e atribuição

O código-fonte original deste projeto é disponibilizado sob a licença MIT, conforme o arquivo [`LICENSE`](LICENSE). A licença do código não substitui os termos aplicáveis ao conjunto de dados.

A fonte imediata do arquivo `WineQT.csv`, a referência do conjunto original de Cortez et al. (2009) e as informações de atribuição estão documentadas em [`DATASET_ATTRIBUTION.md`](DATASET_ATTRIBUTION.md).

O Pickle deve ser carregado somente quando foi criado pelo próprio projeto ou por uma fonte confiável, pois sua desserialização pode executar código.

Projeto acadêmico desenvolvido para o Tech Challenge da Pós Tech.
