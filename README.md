# Análise de Clustering com Vários Algoritmos

Este projeto realiza a análise de clustering utilizando diferentes algoritmos de agrupamento, incluindo DBSCAN, K-Means e Agglomerative Clustering (AGNES). O código lê um conjunto de dados de um arquivo CSV e executa as análises, gerando coesão, separação, entropia e coeficiente de silhueta para cada algoritmo.

## Dependências

As seguintes bibliotecas Python são necessárias para executar este código:

- `math`
- `copy`
- `itertools`
- `collections`
- `numpy`
- `pandas`
- `matplotlib`
- `sklearn`

## Leitura dos Dados

Os dados são lidos a partir de um arquivo CSV localizado no caminho especificado. A função `naoseimexercomdataframe` é utilizada para extrair as coordenadas X e Y, assim como os rótulos dos dados.

## Funções Principais

### `mean(v, labels, dbscan=False)`

Calcula a média de um vetor `v` com base nos rótulos fornecidos. Se o parâmetro `dbscan` for `True`, o cálculo desconsidera o rótulo -1.

### `cohesion(lx, ly, lb)`

Calcula a coesão dos clusters, levando em consideração as coordenadas X (`lx`), Y (`ly`) e os rótulos dos clusters (`lb`).

### `separation(lx, ly, lb)`

Calcula a separação entre os clusters com base nas coordenadas e rótulos fornecidos.

### `entropy(origLabels, clusteringLabels)`

Calcula a entropia dos rótulos originais em relação aos rótulos gerados pelos algoritmos de clustering.

## Algoritmos de Clustering

### DBSCAN

O algoritmo DBSCAN é executado com parâmetros definidos (`eps=0.9`, `min_samples=20`). O resultado é visualizado em um gráfico, comparando os clusters gerados com os rótulos originais. As métricas de coesão, separação, entropia e coeficiente de silhueta são calculadas e impressas.

### K-Means

O algoritmo K-Means é aplicado com um número definido de clusters (`n_clusters=9`, `max_iter=27`). Os resultados também são plotados, permitindo a comparação com os rótulos originais. As mesmas métricas são calculadas e apresentadas.

### AGNES (Agglomerative Clustering)

O algoritmo AGNES é executado com o parâmetro de número de clusters definido (`n_clusters=3`, `linkage="single"`). Os resultados são visualizados, e as métricas de coesão, separação, entropia e coeficiente de silhueta são calculadas e exibidas.

## Resultados

As saídas do código incluem a coesão, separação e entropia para cada algoritmo, assim como o coeficiente de silhueta, permitindo uma comparação entre os métodos de clustering.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para fazer um fork deste repositório e enviar um pull request com melhorias ou correções.

## Licença

Este projeto está sob a licença MIT.
