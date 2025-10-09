# Big-Data-Analytics-Project

## E-commerce Consumer Behavior Analysis

This project analyzes e-commerce consumer behavior using PySpark and provides product recommendations using multiple approaches.

### Features

#### Recommendation Systems

The notebook includes three recommendation approaches:

1. **ALS (Alternating Least Squares)** - Matrix Factorization approach
   - Uses PySpark's MLlib ALS implementation
   - Scalable for large datasets
   - Discovers latent factors in user-item interactions
   
2. **User-Based Collaborative Filtering (CF)** - Memory-based approach
   - Finds similar users based on purchase patterns
   - Uses cosine similarity between user vectors
   - Recommends items that similar users have purchased
   - Explainable recommendations
   
3. **Item-Based Collaborative Filtering (CF)** - Memory-based approach
   - Finds similar items based on user purchase patterns
   - Uses cosine similarity between item vectors
   - Recommends items similar to what the user has already purchased
   - Better stability with sparse data

### How to Use

1. Open the notebook in Google Colab
2. Upload your e-commerce dataset CSV file(s)
3. Run the cells sequentially
4. Compare recommendation results from different approaches

### Requirements

- PySpark 3.5.1
- PyArrow
- Pandas
- Matplotlib
- Seaborn

### Dataset

The notebook expects e-commerce consumer behavior data with at minimum:
- Customer ID
- Product Category
- Purchase Amount (optional)
- Other behavioral features

### Outputs

- Customer segmentation analysis
- Purchase pattern visualizations
- Product recommendations using multiple approaches
- Category performance metrics
- Downloadable CSV reports