# 🎬 MovieLens 100k Recommendation System

A movie recommendation system built using the **MovieLens 100k dataset**. This project demonstrates different collaborative filtering techniques and matrix factorization to suggest movies to users.

---

## 📌 Features

* **User-Based Collaborative Filtering** → Recommends movies based on similar users’ preferences.
* **Item-Based Collaborative Filtering** → Suggests movies similar to a selected movie.
* **Matrix Factorization (SVD)** → Predicts unseen ratings and recommends top movies.

---

## 📂 Dataset

The project uses the **MovieLens 100k dataset**.

* **Ratings file:** `u.data`

  * Columns: `user_id`, `item_id`, `rating`, `timestamp`
* **Movies file:** `u.item`

  * Columns: `movie_id`, `title`, `release_date`, `IMDb_URL`, genres (binary flags)

🔗 [MovieLens Dataset](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)

---

## ⚙️ Installation & Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/MovieLens-Recommender.git
   cd MovieLens-Recommender
   ```

2. Install dependencies:

   ```bash
   pip install pandas numpy scikit-learn scipy
   ```

3. Download and extract the **MovieLens 100k dataset** inside the project folder.

4. Run the recommender system:

   ```bash
   python recommender.py
   ```

---

## 🚀 Code Overview

### 1. Load Dataset

```python
ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
movies = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", header=None,
                     names=["movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
                            "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
                            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])
data = pd.merge(ratings, movies, left_on="item_id", right_on="movie_id")
```

### 2. User-Based Collaborative Filtering

```python
recommend_movies_user(1)
```

➡️ Example Output:

```
Walk in the Clouds, A (1995)    5.0
Waiting for Guffman (1996)      5.0
American in Paris, An (1951)    5.0
Titanic (1997)                  5.0
Sophie's Choice (1982)          5.0
```

### 3. Item-Based Collaborative Filtering

```python
recommend_movies_item("Toy Story (1995)")
```

➡️ Example Output:

```
Star Wars (1977)                 0.734572
Return of the Jedi (1983)        0.699925
Independence Day (ID4) (1996)    0.689786
Rock, The (1996)                 0.664555
Mission: Impossible (1996)       0.641322
```

### 4. Matrix Factorization (SVD)

```python
recommend_movies_svd(1)
```

➡️ Example Output:

```
E.T. the Extra-Terrestrial (1982)         3.569155
Batman (1989)                             3.212662
Dave (1993)                               3.086038
Ulee's Gold (1997)                        2.939371
One Flew Over the Cuckoo's Nest (1975)    2.650993
```

---

## 📊 Techniques Used

* **Cosine Similarity** → Measures similarity between users/movies.
* **Collaborative Filtering** → Uses ratings from similar users/items.
* **SVD (Singular Value Decomposition)** → Matrix factorization to capture latent features.

---

## 📝 Results

✅ Successfully generates movie recommendations using three different methods:

* User-Based → Finds users with similar tastes.
* Item-Based → Finds movies with similar rating patterns.
* SVD → Predicts unseen ratings using latent features.

---

## 🔮 Future Improvements

* Add **hybrid models** combining CF + SVD.
* Use **deep learning approaches** (Neural CF, Autoencoders).
* Implement a **web interface** with Flask/Streamlit.
* Optimize similarity measures with **Pearson correlation**.

---

## 👨‍💻 Author

* **Your Name**
* GitHub: [@touseef7878](https://github.com/touseef7878)

---

## 📜 License

This project is licensed under the MIT License.
