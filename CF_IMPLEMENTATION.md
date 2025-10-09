# Collaborative Filtering Implementation (Without ALS)

## Overview

This document describes the Collaborative Filtering (CF) implementation added to the Ecommerce Consumer Behavior notebook. The implementation provides recommendation functionality without using ALS (Alternating Least Squares).

## Implementation Details

### Location
- **Notebook Cell**: Cell 9 (inserted after the existing ALS cell)
- **File**: `Ecommerce_Consumer_Behavior.ipynb`

### Approaches Implemented

#### 1. User-Based Collaborative Filtering
**Concept**: "Users who are similar to you also liked..."

**Algorithm**:
1. Build user-item interaction matrix from purchase data
2. Calculate user similarity using cosine similarity:
   - For each pair of users, find items both have purchased
   - Compute cosine similarity of their rating vectors
3. For each user, find their most similar users
4. Recommend items that similar users have purchased but the target user hasn't

**PySpark Implementation**:
- Self-join on items to find user pairs with common purchases
- Aggregate to compute cosine similarity: `dot_product / (norm1 * norm2)`
- Filter weak similarities (threshold > 0.1)
- Generate recommendations using weighted ratings from similar users

#### 2. Item-Based Collaborative Filtering
**Concept**: "Items similar to what you liked..."

**Algorithm**:
1. Build item-user interaction matrix (transposed)
2. Calculate item similarity using cosine similarity:
   - For each pair of items, find users who purchased both
   - Compute cosine similarity of their rating vectors
3. For items a user has purchased, find similar items
4. Recommend similar items the user hasn't purchased yet

**PySpark Implementation**:
- Self-join on users to find item pairs with common purchasers
- Aggregate to compute cosine similarity
- Track number of common users for quality assessment
- Generate recommendations using weighted ratings from similar items

### Key Features

1. **No Training Phase**: Works directly with the interaction matrix
2. **Explainable**: Can show why items are recommended (similar users/items)
3. **Memory-Based**: Stores and uses all interaction data
4. **Threshold Filtering**: Removes weak similarities (< 0.1) to improve quality
5. **Anti-Join**: Prevents recommending items users have already purchased
6. **Top-K Selection**: Returns top 5 recommendations per user

### Similarity Calculation

**Cosine Similarity Formula**:
```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Where:
- `A · B` = dot product (sum of element-wise multiplication)
- `||A||` = magnitude of vector A = sqrt(sum of squares)
- `||B||` = magnitude of vector B = sqrt(sum of squares)

**Range**: 0 to 1 (1 = identical preferences, 0 = no similarity)

### Comparison with ALS

| Aspect | CF (Memory-Based) | ALS (Model-Based) |
|--------|-------------------|-------------------|
| **Training** | No training needed | Requires model training |
| **Explainability** | High - can show similar users/items | Low - uses latent factors |
| **Scalability** | Good for small-medium data | Excellent for large data |
| **Cold Start** | Challenging for new users/items | Similar challenges |
| **Prediction Speed** | Slower for large datasets | Fast after training |
| **Memory Usage** | Stores full interaction matrix | Stores factor matrices |

### Handling Sparse Data

The implementation includes error handling for sparse datasets:
- Try-catch blocks around recommendation generation
- Informative messages if recommendations cannot be generated
- Suggestion to use popularity-based fallback

### Visualization

The implementation includes a bar chart showing the most frequently recommended categories from Item-Based CF, helping to understand recommendation patterns.

## Usage

The CF implementation runs automatically when the notebook cell is executed, provided the required columns (`customer_id` and `category`) are present in the dataset.

**No code changes needed** - simply run Cell 9 after the data preprocessing cells.

## Output

The cell produces:
1. Interaction matrix sample
2. User similarity matrix (top similar user pairs)
3. User-Based CF recommendations (top 5 per user)
4. Item similarity matrix (top similar item pairs)  
5. Item-Based CF recommendations (top 5 per user)
6. Visualization of most recommended categories
7. Comparison summary of CF vs ALS

## Testing

Run `validate_cf.py` to verify the CF logic:
```bash
python3 validate_cf.py
```

This validates:
- Cosine similarity calculations
- User-based similarity logic
- Item-based similarity logic
- Recommendation generation logic
