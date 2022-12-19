# LoL Champion Embedding

League of legend champion embedding with similarity learning based on user preference

## Method
### 1. Get summoner ids with crawling

From op.gg ranking page with BeautifulSoup

### 2. Save the summoner's unique id and mastery point in DB

Used RiotAPI and SQLite

### 3. Train champion embedding based on the similarity score

When a player plays champions for many times, we consider them to be similar.

### 4. Visualize champions

Visualize the embedding vector of each champion by dimensionality reduction method (t-SNE showed reliable result)

## Result

Visualization with TSNE:
![champion_clustering_tsnr_kr](./result/champion_clustering_tsne_kr.png)

## Reference

[WillKoehrsen's work](https://github.com/WillKoehrsen/wikipedia-data-science/blob/master/notebooks/Book%20Recommendation%20System.ipynb)

[giantZorg's work](https://github.com/giantZorg/Lol_champion_embeddings)
