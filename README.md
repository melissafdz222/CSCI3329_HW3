# CSCI 3329 - Homework 3 Report

### 1. Dataset Description

- **Name**: [First-Order Theorem Proving Dataset](https://archive.ics.uci.edu/dataset/249/first+order+theorem+proving) 
- **Source**: UCI ML Repository
- **Number of samples**: 6,118 total (3,059 train, 1,529 validation, 1,530 test)
- **Number of features**: 51 (13 static + 38 dynamic features)
- **Number of classes**: 6 (5 heuristics + decline option)

**Class Distribution:**
| Class | Description | Count | Percentage |
|-------|-------------|-------|------------|
| H0 | Decline (no proof) | 2,554 | 41.7% |
| H1 | Heuristic 1 | 1,089 | 17.8% |
| H2 | Heuristic 2 | 486 | 7.9% |
| H3 | Heuristic 3 | 748 | 12.2% |
| H4 | Heuristic 4 | 617 | 10.1% |
| H5 | Heuristic 5 | 624 | 10.2% |

### 2. Preprocessing

- **Missing values**: None reported in dataset documentation
- **Irrelevant columns**: Removed source column after splitting
- **Encoding**: Target converted to single label (0-5)
- **Scaling**: StandardScaler (zero mean, unit variance) for KNN and MLP

### 3. Part 2 - Algorithm Comparison

| Algorithm | Mean Accuracy | Std Dev |
|-----------|---------------|---------|
| Perceptron | 0.3521 | 0.0424 |
| Logistic Regression | 0.4654 | 0.0269|
| KNN | 0.5300 | 0.0262  |
| Gaussian NB | 0.1638 | 0.0209 | 
| Neural Network | 0.4720 | 0.0287 |

### 4. Part 3 - Feature Selection

**Search Method**: Forward Selection (51 features > 15)

| Algorithm | Best Feature Subset | Mean Accuracy | Std |
|-----------|---------------------|---------|-----|
| Perceptron | 10 | 0.3245 | 0.0504 |
| Logistic Regression | 10 | 0.4568 | 0.0197 |
| KNN | 12 | 0.5284 | 0.0194 | 
| Gaussian NB | 10 | 0.3799 | 0.0273 |
| Neural Network | 7 | 0.4354 | 0.0235 |

### 5. Discussion
**Part 2 vs Part 3 Comparison**
  Logistic Regression showed greater improvement with the highest increase in accuracy, after reducing features from 51 to 19. Gaussian Naive Bayes also benefited beacause removing features in the way helped for better estimates. The Perceptron showed a small change which also simplified some parts. K-Nearest Neighbors demonstrated also a slight improvement, as fewer features reduced some of the impact on the calculations. Finally, Neural Network was the only algorithm that stayed pretty much the same based on the data. Overall, feature selection helped all of the algorithms, while also reducing feature counts by 50-70%.

### 6. Reproduction

```bash
#Python version
python --version  # 3.10+

#Install dependencies
pip install -r requirements.txt

#Run experiment
python homework3.py
