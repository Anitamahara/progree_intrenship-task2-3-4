# Task 3: Customer Segmentation (K-Means + PCA)

**Tools:** Python, Scikit-Learn, Pandas, Matplotlib

## Method
1. Standardized features with StandardScaler
2. Reduced dimensions with PCA (2 components, 93.5% variance retained)
3. Chose k=5 using the Elbow Method (elbow.png)
4. Fitted K-Means; Silhouette score = 0.747

## Segments
| Segment | Profile |
|---|---|
| 0 | Premium spenders |
| 1 | Budget shoppers |
| 2 | Young low-spenders |
| 3 | Affluent loyal buyers |
| 4 | Average customers |

## Run
pip install pandas scikit-learn matplotlib
python clustering.py

## Output
elbow.png, segments.png, segment_profile.csv, customers_segmented.csv