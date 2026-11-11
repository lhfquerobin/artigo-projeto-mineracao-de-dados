# Análise dos Fatores Associados à Receita de Filmes

Equipe: Luiz Henrique Floriano Querobin, Sara Ferreira Da Silva e Kauã Ferreira Rocha.
Metodologia: CRISP-DM. Fontes: TMDB_all_movies.csv, Kaggle (Alan Vourc’h), e arquivos públicos title.basics e title.ratings do IMDb.

## Pergunta
Quais características de um filme estão mais relacionadas ao seu sucesso financeiro e é possível utilizar essas características para estimar sua receita?

O alvo é receita nominal registrada, sem equivalência a lucro. Popularidade, notas e votos entram somente na exploração. Os modelos usam orçamento, duração, idioma, gêneros, ano e mês de lançamento.

## Preparação e execução no Windows
Instale Python 3.12 e execute na raiz do projeto:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src/baixar_imdb.py
.venv\Scripts\python.exe src/explorar_imdb.py
.venv\Scripts\python.exe src/treinar.py
.venv\Scripts\python.exe src/graficos_complementares.py
```

Coloque TMDB_all_movies.csv em data/raw ou na raiz. Fonte: https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates
Não publique os arquivos brutos no GitHub. O hash do TMDB está em results/audit.json. Os hashes, tamanhos e datas dos arquivos IMDb estão em results/imdb/download_manifest.json e results/imdb/audit_imdb.json. Os downloads públicos são atualizados. Uma execução com outra fotografia pode produzir contagens diferentes.

## Ampliação da exploração inicial com IMDb
No seminário, o professor considerou pequena a base do TMDB obtida no Kaggle, que seria nossa única fonte. Acrescentamos o IMDb para ampliar a exploração com filmes, séries, minisséries e episódios. Mantivemos o TMDB como fonte da receita usada na regressão. Os arquivos são lidos compactados, em blocos de 100 mil linhas. O script registra ausências, valida campos numéricos, sinaliza anos futuros e distingue ausência estrutural de endYear em títulos que não são séries. A auditoria global de chaves acumula apenas sufixos numéricos de identificadores, sem manter todas as linhas na memória.

As bases são perfiladas separadamente: ratings são avaliações de títulos, não novos filmes, e o cruzamento por identificador evita somar sobreposições como observações independentes. O subconjunto IMDb de tipo movie é cruzado com o TMDB para uma referência exploratória salva em data/processed/tmdb_imdb_movies_reference.csv. Esse arquivo não entra no treinamento e não muda o filtro da amostra financeira. Séries e episódios aparecem nas estatísticas iniciais e ficam fora dessa referência de filmes.

O script verifica por hashes que métricas, previsões, auditoria financeira e modelo existentes não foram alterados. Esta etapa amplia o tratamento inicial, mas não amplia os exemplos com receita conhecida. Os arquivos brutos e a referência linha a linha não entram no ZIP do repositório. Para economizar memória, processa-se um bloco por vez. O script mantém identificadores compactos e as correspondências com o TMDB para auditoria e integração.

Execução realizada em 18/10/2026: 12.872.093 títulos em basics, incluindo 759.771 movie, 307.146 tvSeries, 73.667 tvMiniSeries e 9.955.259 tvEpisode. O arquivo ratings contém 1.722.371 linhas. O cruzamento encontrou 690.356 linhas do TMDB com metadados IMDb e 385.149 com tipo movie. Foram tratados quatro valores inválidos de duração e sinalizados 4.627 anos futuros. As demais regras de validação são registradas mesmo quando não encontram inconsistências.

A amostra financeira original contém 21.935 correspondências movie, 2.797 sem correspondência e 1.237 de outros tipos IMDb, incluindo 29 séries/minisséries e 19 episódios. Esse resultado revela heterogeneidade já existente no TMDB. Nenhum desses exemplos foi acrescentado pela nova etapa. A referência movie não foi usada para refiltrar o treinamento, conforme o escopo de preservar os experimentos.

A primeira rodada pode ser reproduzida com `python src/treinar_inicial.py`, salvando em results/initial. A rodada revisada salva em results. Execuções substituem os resultados correspondentes, então preserve uma cópia antes de testar novos dados. Pequenas diferenças numéricas podem ocorrer entre plataformas e versões.

## Resultados executados
- Base: 1.260.483 registros; 25.969 após o recorte analítico.
- Treino: 1905–2018, 18.862 títulos.
- Validação: 2019–2022, 3.271 títulos.
- Teste: 2023–2025 (até 18/10/2025), 3.836 títulos.
- Rodada inicial: Ridge selecionada por MAE de validação, com instabilidade no teste (R² -37,12).
- Revisão: logaritmo também na duração; Random Forest selecionada na validação.
- Teste revisado: MAE 15.540.574,80; RMSE 84.042.763,55; R² 0,2597.
- Referência no teste: MAE 20.684.673,53. Redução de MAE: 24,87%.

A revisão ocorreu após inspeção do teste inicial. Os resultados revisados são exploratórios, sem confirmação independente. A cobertura financeira é incompleta e os valores não foram corrigidos pela inflação. Não houve classificação de sucesso por nota, reamostragem de classes nem uso de receita como entrada. O IMDb foi integrado somente na exploração e no tratamento complementar.

## Arquivos
- src/graficos_complementares.py: cobertura dos atributos, receita por período e erro por quartil, a partir dos resultados salvos.
- src/treinar.py: preparação, validação, seleção, avaliação, gráficos e auditoria da revisão.
- src/treinar_inicial.py: versão inicial preservada.
- src/prever.py: consulta local ao modelo gerado.
- src/baixar_imdb.py: download dos dois arquivos compactados necessários.
- src/explorar_imdb.py: tratamento em blocos e cruzamento exploratório por identificador.
- results/: métricas, previsões, importância por permutação, gráficos, auditoria e primeira rodada.
- results/imdb/: contagens por tipo, ausências antes/depois, cobertura do cruzamento, manifestos e auditoria da preservação do modelo.
- artigo/: artigo final em PDF.
- tools/conferir_resultados.py: auditoria das métricas, identificadores e separação temporal.

O treinamento salva o modelo em results/model.joblib. Esse arquivo é gerado localmente e não é versionado. Para consultar após treinar:

```powershell
python src/prever.py --budget 10000000 --runtime 110 --genres "Drama, Romance" --language en --year 2025 --month 6
```
