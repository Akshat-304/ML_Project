# Automated Music Playlist Generator

## Project Overview
With the explosive growth of music streaming platforms such as Spotify, Apple Music, and Wynk, curating personalized playlists for users has become increasingly complex. This project aims to automate the categorization of songs into playlists by leveraging machine learning techniques. Unlike conventional methods that rely solely on audio features, this project integrates **lyrical analysis** alongside **audio-based features** to enhance playlist generation accuracy.

## Features and Methodologies
- **Dataset:** GTZAN Music Genre Dataset (10 genres, 100 audio samples per genre, 30-second waveform format).
- **Feature Extraction:**
  - Chroma Features
  - MFCC (Mel-Frequency Cepstral Coefficients)
  - Spectral Centroid, Bandwidth, Rolloff
  - Zero Crossing Rate, RMS Energy, Tempo, Harmony, Percussive Features
- **Dimensionality Reduction:**
  - PCA (Principal Component Analysis) for feature selection.
  - t-SNE (t-distributed Stochastic Neighbor Embedding) for high-dimensional data visualization.
- **Machine Learning Models:**
  - **Classification:** Decision Trees, SVM (RBF Kernel), MLP Classifier, XGBoost, LightGBM, Random Forest, AdaBoost.
  - **Clustering:** K-Means with optimal cluster selection via Elbow Method and Silhouette Scores.
- **Performance Evaluation:**
  - Precision, Recall, F1-Score, and Accuracy
  - GridSearchCV for hyperparameter tuning and cross-validation.

## Results
- **Best Performing Models:**
  - **Before Hyperparameter Tuning:** MLP and Random Forest achieved **78.5% accuracy**.
  - **After Hyperparameter Tuning:** LightGBM and Random Forest achieved **77% accuracy**.
- **Feature Analysis:**
  - Audio-based features significantly contribute to genre classification.
  - Lyrical analysis enhances classification by adding thematic structure.

## Installation & Usage
### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Required Libraries: `librosa`, `scikit-learn`, `xgboost`, `lightgbm`, `numpy`, `matplotlib`, `pandas`

### Steps to Run the Project
1. Clone the repository:
   ```sh
   git clone https://github.com/Kartikeya2022241/ML_final_project
   cd ML_final_project
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```sh
   jupyter notebook file.ipynb
   ```

## Authors
- **Aahan Piplani** (IIITD)
- **Abhishek Bansal** (IIITD)
- **Akshat Kothari** (IIITD)
- **Kartikeya** (IIITD)
- **Vinayak Agrawal** (IIITD)

For further details, check the full project report [here](ML_project_final.pdf).

