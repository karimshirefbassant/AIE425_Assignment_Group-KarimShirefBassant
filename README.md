# Intelligent Recommender Systems Assignment

## Team
* **Shiref Khaled Elhalawany** - 221100944
* **Karim Ashraf Elsayed** - 221100391
* **Bassant Kamal Mesilam** - 221100244

## Dataset
This project uses the **Amazon Electronics** dataset.
* **Source:** [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/)
* **Dataset:** Electronics ratings only
* **Size:** 20,994,353 ratings
* **Download Link:** [Electronics.csv](https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Electronics.csv)

## Project Overview
This repository contains the implementation of various collaborative filtering techniques and statistical analyses on the Amazon Electronics dataset. The project is divided into three main sections:

### Section 1: Statistical Analysis
*   **Notebook:** `section1_statistical_analysis/statistical_analysis.ipynb`
*   **Description:**
    *   Loads the dataset and performs exploratory data analysis.
    *   Calculates global statistics: total number of users, products, and ratings.
    *   Analyzes rating distributions (min, max, average).
    *   Computes user activity (ratings per user) and item popularity (ratings per item).
    *   Visualizes the long-tail distribution of item ratings.
    *   Categorizes items based on popularity.

### Section 2: Neighborhood-Based Collaborative Filtering
This section implements memory-based collaborative filtering algorithms.

#### Part 1: User-Based CF
*   **Notebook:** `section2_neighborhood_cf/part1_user_based_cf/user_based_cf.ipynb`
*   **Description:**
    *   Implements **User-Based Collaborative Filtering**.
    *   Loads selected target users and their co-rating neighbors.
    *   Calculates **Cosine Similarity** between users based on common ratings.
    *   Identifies the top-k similar users (nearest neighbors).
    *   Predicts ratings for unrated items using weighted averages of neighbors' ratings.
    *   Implements **Discounted Similarity** to penalize neighbors with few common items and improve prediction accuracy.

#### Part 2: Item-Based CF
*   **Notebook:** `section2_neighborhood_cf/part2_item_based_cf/item_based_cf.ipynb`
*   **Description:**
    *   Implements **Item-Based Collaborative Filtering**.
    *   Loads target items and co-rating user data.
    *   Calculates **Adjusted Cosine Similarity** between items (handling user rating bias by subtracting user means).
    *   Identifies top-k similar items.
    *   Predicts ratings for target users on specific items.
    *   Compares standard Item-Based CF with Discounted Similarity approaches.

### Section 3: Clustering-Based Collaborative Filtering
This section explores clustering techniques to group users and items, aiming to improve recommendation scalability and address specific problems like the cold-start.

#### Part 1: User Clustering (Average Ratings)
*   **Notebook:** `section3_clustering_based_cf/part1_user_clustering_avg_ratings/part1_user_clustering_avg_ratings.ipynb`
*   **Description:**
    *   Clusters users based on their **Average Rating** behavior.
    *   Computes user statistics (mean, std dev) and normalizes features (Z-score).
    *   Applies **K-Means Clustering** (1D) with various K values (5, 10, 15, 20, 30, 50).
    *   Analyzes cluster centroids to understand "Generous" vs. "Strict" rater groups.

#### Part 2: User Clustering (Common Ratings)
*   **Notebook:** `section3_clustering_based_cf/part2_user_clustering_common_ratings/part2_user_clustering_common_ratings.ipynb`
*   **Description:**
    *   Clusters users based on their **Co-rating Activity** (overlap with other users).
    *   Features: Average common ratings, Max common ratings, Min common ratings.
    *   Normalizes features manually without built-in scalers.
    *   Applies **K-Means Clustering** to segment users by their connectivity in the social/rating graph.
    *   Evaluates clusters using **Silhouette Scores** and Elbow method to find optimal K.

#### Part 3: Item Clustering (Average Raters)
*   **Notebook:** `section3_clustering_based_cf/part3_item_clustering_avg_raters/part3_item_clustering_avg_raters.ipynb`
*   **Description:**
    *   Clusters items based on popularity and rating patterns.
    *   Features: Number of raters, Average rating, Standard deviation of ratings.
    *   Normalizes item features using Z-scores.
    *   Applies **K-Means Clustering** to group items (e.g., "Popular & High Rated", "Niche & Mixed Ratings").
    *   Determines optimal K using WCSS (Elbow Method) and Silhouette analysis.
    *   Analyzes characteristics of resulting item clusters (Popular vs. Long-tail).

#### Part 4: Cold-Start Problem
*   **Notebook:** `section3_clustering_based_cf/part4_cold_start_clustering/part4_cold_start_clustering.ipynb`
*   **Description:**
    *   Simulates a **Cold-Start Scenario** by hiding a percentage of ratings for selected test users/items.
    *   Develops a strategy to assign new/cold-start users to existing clusters based on limited initial data.
    *   Generates recommendations for cold-start users using cluster-based neighbors.
    *   Evaluates performance (MAE, RMSE) of the cold-start strategy against ground truth data.

## Requirements
The project relies on standard Python data science libraries.
*   `pandas`
*   `matplotlib`
*   `numpy`
*   `random`
*   `csv`
*   `math`

See `requirements.txt` for the list.


## How to Run
1. Ensure the dataset `Electronics.csv` is placed in the `dataset/` directory (or update paths accordingly).
2. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate      # On Linux/Mac
    venv\Scripts\activate         # On Windows
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the notebooks in the order presented to reproduce the analysis and results.
