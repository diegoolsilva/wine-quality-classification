# Atribuição e licença do conjunto de dados

## Arquivo utilizado

- **Arquivo:** `data/raw/WineQT.csv`
- **Fonte imediata da cópia utilizada:** M Yasser H, *Wine Quality Dataset*, Kaggle: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
- **Conteúdo:** 1.143 registros com 11 variáveis físico-químicas, a nota `quality` e o identificador `Id`.

A página Kaggle é citada como a fonte imediata do arquivo `WineQT.csv` utilizado neste projeto. A base corresponde à família de dados *Wine Quality* documentada pelo UCI Machine Learning Repository. A referência e a licença dos dados originais são mantidas abaixo para preservar a atribuição aos autores.

## Referência do conjunto original

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Wine Quality* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T

- Página do conjunto no UCI: https://archive.ics.uci.edu/dataset/186/wine+quality
- Licença indicada pelo UCI: **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
- Texto da licença: https://creativecommons.org/licenses/by/4.0/

A CC BY 4.0 permite compartilhar e adaptar o material, inclusive para fins comerciais, desde que seja fornecida atribuição apropriada, seja incluído o link da licença e sejam indicadas alterações realizadas. Consulte a página oficial do UCI e o texto da licença para os termos completos.

## Alterações e tratamento realizados neste projeto

O arquivo bruto `WineQT.csv` é mantido em `data/raw/`. Durante o processo de análise e modelagem, o projeto:

1. exclui `Id` das variáveis preditoras;
2. cria a variável binária `target`, com valor 1 para `quality >= 7` e 0 para `quality < 7`;
3. identifica perfis físico-químicos repetidos e, para a base utilizada na modelagem, mantém uma ocorrência por perfil;
4. cria as variáveis derivadas `free_sulfur_ratio` e `volatile_to_fixed_acidity`.

Essas operações são transformações analíticas realizadas pelo projeto; não modificam o arquivo bruto distribuído em `data/raw/`.

## Observação sobre a licença do projeto

A licença MIT encontrada no arquivo `LICENSE` cobre o código-fonte original deste repositório. Ela **não substitui** a licença do dataset nem altera os direitos dos autores originais. A atribuição e os termos aplicáveis aos dados devem ser preservados quando o arquivo for redistribuído.
