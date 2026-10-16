# Analyzing Artist Styles Using K-means Clustering

I test multiple K-means models with varying values of K to learn hidden clusters in song data from my favorite artists. To find the optimal value of K, I compare the final scores of the K-means objective function for each model:

```math
J = \sum_{i=1}^{k} \sum_{x \in C_i} \Vert{}x - \mu_i\Vert{}^2
```
\
Let mu denote the set of centroids, and C represent the function that assigns points x to centroids. 

(Summary of results here)

(Blog post forthcoming)

For future work, I will develop a model to run a "favorite song tournament" that takes in a song dataset provided by the user (8-128 songs) and the number of songs in each match (2-8). It will prompt the user to enter the results for 1 round of the tournament, and use a feed-forward neural network to analyze the importance of each feature in the dataset in deciding match outcomes. Using that learned model, it will automatically complete the rest of the matches and display the results of the tournament. 

---

## Experimental Design

In this experiment, I will first empirically test different K-values to determine the optimal K-means model to use for clustering the songs. Each value of K will receive 10 trials of random centroid initializations, and the final positions of the centroids will be averaged across the 10 trials to arrive at the final centroids for each K-value. Then, I will score each K-value by computing the within cluster sum of squares distance using the euclidean distance formula. The K-value with the smallest score is chosen. 

Once I use the chosen model to cluster the songs, I will analyze the distribution of artists represented in each cluster (number of their songs assigned to that cluster), as well as the distribution of clusters touched by each artist (are they mostly in one, a few, or scattered over many, which measures the versatility of their craft). Lastly, I will comment on the distance between the clusters. I expect the largest K-value to have the smallest score on the objective function, but that is not necessarily the best model. If some cluster centroids are very close together, they can effectively be merged. 

---

## Explanation of Raw Data

The dataset, artist_data.xlsx, contains 167 songs collectively written by 7 artists (Sabrina Carpenter, Gracie Abrams, Olivia Rodrigo, Erin LeCount, Tate McRae, Against The Current, and Crown The Empire). Each observation has the song name, an artist label, album label, and scores for energy, positivity, and introspectiveness. Here is the specification for that data:

| Score | Energy | Positivity | Introspectiveness |
|---|---|---|---|
| 0 | Slow, whispery | Hopelessness | Singular external |
| 1 | Slow acoustic | Anger | Group external |
| 2 | Slow rock ballad | Annoyance | Society |
| 3 | Moderate pop | Bittersweet | Social norms |
| 4 | Classic rock | Hopeful | Group internal |
| 5 | Fast rock | Redemption | Couple internal |
| 6 | Fast punk | Love | Habit analysis |
| 7 | Fast metal | Elation | Fully internal |

### Prediction
I stored the data in a google sheet, and used Gemini to analyze trends to gain some level of intuition before running K-means. 
\
(put in pictures of tables)

###
---

## Results & Analysis

Here were my results from running the full experiment:

### 1. Optimal Value of K

(Results here)

![Elbow Curve: Objective Function Score vs K](plots/elbow_curve.png)
![All 8 K-Means Models Comparison Side-by-Side (K=7 to K=14)](plots/all_kmeans_models_3d.png)
 
---

### 2. Distribution of Clusters

(Comment on which models space out centroids more than others)

![Cluster Centroid Distribution](plots/song_clusters_3d.png)
 
---

### 3. Distribution of Artists' Songs

![Songs by Artist (3D Feature Space)](plots/songs_by_artist_3d.png)
![Cluster Centroids vs. Artist Average Coordinates](plots/centroids_vs_artists_3d.png)

(Compare centroids to average scores of each feature for each artist)

---

### 4. Versatility 

To quantify the versatility of an artist's craft, I compute a versatility score for each artist based on the Gini Impurity of their song cluster assignments:

```math
I_G = 1 - \sum_{i=1}^{k} p_i^2
```
\
Where $k$ denotes the number of clusters, and $p_i$ represents the proportion of an artist's songs assigned to cluster $i$. A score of 0 indicates that all songs belong to a single cluster (highly focused style), whereas higher values indicate songs scattered across multiple distinct clusters (greater versatility).

| Artist | Song Count | Clusters Touched | Gini Impurity | Normalized Gini |
|---|:---:|:---:|:---:|:---:|
| Against The Current | 19 | 10 | 0.8753 | 0.9427 |
| Olivia Rodrigo | 24 | 9 | 0.8611 | 0.9274 |
| Sabrina Carpenter | 25 | 9 | 0.8416 | 0.9063 |
| Tate McRae | 27 | 9 | 0.8368 | 0.9011 |
| Gracie Abrams | 32 | 9 | 0.8281 | 0.8918 |
| Erin LeCount | 12 | 6 | 0.7778 | 0.8376 |
| Crown The Empire | 28 | 6 | 0.7168 | 0.7720 |

(Compare centers of each artists' first and last album, and analyze the distribution of centroid assignments for their songs)

---

### 5. Conclusions & Discussion

(Compare results to predictions)
(Discuss possible reasons for discrepancies)
(Comment on anything that changed from the original specification)
(Mention future work)

---

## Codebase Architecture

```
kmeans_artists/
├── preprocess.py             # Load data (features are already on the same scale, keep that scale for interpretability / plotting)
├── kmeans.py                 # The K-means algorithm, run for a given value of K
├── stats.py                  # Will analyze the data and compute metrics for stages 3 and 4 of results
├── experiment.py             # Constructs the full experiment, including outputs
├── runner.py                 # CLI interface for custom trials, rapid smoke tests, and full experiment run
├── Kmeans_SongClusters.pdf   # Original mathematical problem specification
├── artist_data.xlsx          # Original dataset 
├── plots/                    # Directory for generated experiment visualizations
│   ├── elbow_curve.png       # Objective function score (WCSS) vs K
│   ├── all_kmeans_models_3d.png # Side-by-side 3D scatter plots for all 8 K-means models (without centroids)
│   ├── song_clusters_3d.png  # 3D scatter plot of songs colored by cluster with centroids
│   ├── songs_by_artist_3d.png# 3D scatter plot of songs colored by artist
│   └── centroids_vs_artists_3d.png # 3D plot comparing cluster centroids and artist average coordinates
├── tests/                    # Automated unit and integration test suite
│   ├── test_models.py        # Tests the logic of the K-means models
│   ├── test_math.py          # Ensure that all of the calculations are mathematically correct
│   ├── test_experiment.py    # Ensure that the pipeline runs correctly and outputs are in the correct format
│   └── test_smoke.py         # Rapid end-to-end integration smoke test
└── README.md                 # Project specification, experiment documentation, architecture, and usage
```

---

## Installation & Usage

### Prerequisites
```bash
pip install pandas openpyxl numpy scipy matplotlib
```

### Run unittests
Run the complete automated test suite (16 unit and integration tests):
```bash
python3 -m unittest discover -s tests
```

### Run a quick smoke test
Verify that the experiment runs end-to-end with a rapid integration test:
```bash
python3 runner.py --smoke-test
```

### Run the experiment

```bash
python3 runner.py
```

### Command-Line Arguments Reference
| Argument | Description | Default |
|---|---|---|
| `--smoke-test` | Run a rapid end-to-end integration check across both phases | `False` |
| `--trials N` | Number of replications of each K-means model | `10` |
| `--minK N` | Smallest K-value for K-means | `7` |
| `--maxK N` | Largest K-value for K-means | `14` |
| `--output PATH` | Path to export plots | `None` |
