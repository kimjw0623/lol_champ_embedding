# LoL Champion Embedding

League of legend champion embedding with similarity learning based on user preference

## Method
### Get summoner ids with crawling

From op.gg ranking page with BeautifulSoup

### Save summoner's unique id and mastery point as DB

Used RiotAPI 

### Train champoin embedding based on similarity score

When a player plays champions for many times, we consider them to be similar.

## Result

Visualization with TSNE:
![champion_clustering_tsnr_kr](./result/champion_clustering_tsne_kr.png)

## Reference

[WillKoehrsen's work](https://github.com/WillKoehrsen/wikipedia-data-science/blob/master/notebooks/Book%20Recommendation%20System.ipynb)

[giantZorg's work](https://github.com/giantZorg/Lol_champion_embeddings)