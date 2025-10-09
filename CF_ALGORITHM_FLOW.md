# Collaborative Filtering Algorithm Flow

## User-Based Collaborative Filtering

```
Step 1: Build User-Item Interaction Matrix
┌─────────────────────────────────────────┐
│ User  │ Item_A │ Item_B │ Item_C │ ...  │
├───────┼────────┼────────┼────────┼──────┤
│ U1    │  5.0   │  3.0   │  4.0   │ ...  │
│ U2    │  4.5   │  3.5   │  4.5   │ ...  │
│ U3    │  2.0   │  5.0   │   -    │ ...  │
└─────────────────────────────────────────┘

Step 2: Calculate User Similarity (Cosine Similarity)
         Find users with common purchased items
         
         Similarity(U1, U2) = dot(U1, U2) / (||U1|| × ||U2||)
         
         Example: U1 and U2 both purchased Item_A, Item_B, Item_C
         → High similarity (0.99)

Step 3: Find Similar Users
┌───────────────────────┐
│ For User U1:          │
│ - Similar User: U2    │
│   Similarity: 0.99    │
│ - Similar User: U3    │
│   Similarity: 0.75    │
└───────────────────────┘

Step 4: Generate Recommendations
         For User U1:
         - Look at what U2 purchased: Item_D, Item_E
         - U1 hasn't purchased Item_D or Item_E
         - Score = Similarity(U1, U2) × Rating(U2, Item)
         
         Recommendations for U1:
         1. Item_D (score: 4.95)
         2. Item_E (score: 3.96)
```

## Item-Based Collaborative Filtering

```
Step 1: Build Item-User Interaction Matrix (Transposed)
┌─────────────────────────────────────┐
│ Item   │ User_1 │ User_2 │ User_3 │ │
├────────┼────────┼────────┼────────┼─┤
│ Item_A │  5.0   │  4.5   │  2.0   │ │
│ Item_B │  3.0   │  3.5   │  5.0   │ │
│ Item_C │  4.0   │  4.5   │   -    │ │
└─────────────────────────────────────┘

Step 2: Calculate Item Similarity (Cosine Similarity)
         Find items purchased by common users
         
         Similarity(Item_A, Item_C) = dot(A, C) / (||A|| × ||C||)
         
         Example: Item_A and Item_C both purchased by U1, U2
         → High similarity (0.99)

Step 3: Find Similar Items
┌─────────────────────────┐
│ For Item_A:             │
│ - Similar Item: Item_C  │
│   Similarity: 0.99      │
│ - Similar Item: Item_B  │
│   Similarity: 0.85      │
└─────────────────────────┘

Step 4: Generate Recommendations
         For User U1 (who purchased Item_A, Item_B):
         - Item_A is similar to Item_C (not purchased)
         - Item_B is similar to Item_D (not purchased)
         - Score = Similarity(purchased, candidate) × Rating(user, purchased)
         
         Recommendations for U1:
         1. Item_C (score: 5.95)
         2. Item_D (score: 4.25)
```

## Key Differences: Memory-Based CF vs ALS

```
┌─────────────────────────┬──────────────────────┬──────────────────────┐
│ Aspect                  │ CF (Memory-Based)    │ ALS (Model-Based)    │
├─────────────────────────┼──────────────────────┼──────────────────────┤
│ Approach                │ Direct similarity    │ Matrix Factorization │
│                         │ calculation          │                      │
├─────────────────────────┼──────────────────────┼──────────────────────┤
│ Training                │ No training          │ Iterative training   │
├─────────────────────────┼──────────────────────┼──────────────────────┤
│ Explainability          │ High - can show      │ Low - uses latent    │
│                         │ similar users/items  │ factors              │
├─────────────────────────┼──────────────────────┼──────────────────────┤
│ Scalability             │ Good for small-      │ Excellent for very   │
│                         │ medium datasets      │ large datasets       │
├─────────────────────────┼──────────────────────┼──────────────────────┤
│ New Users/Items         │ Needs common items   │ Needs training data  │
├─────────────────────────┼──────────────────────┼──────────────────────┤
│ Implementation          │ Cosine similarity    │ Gradient descent/    │
│                         │ computations         │ Least squares        │
└─────────────────────────┴──────────────────────┴──────────────────────┘
```

## Cosine Similarity Formula

```
For vectors A and B:

                    A · B
similarity(A, B) = ─────────
                   ||A|| ||B||

Where:
- A · B = dot product = Σ(A[i] × B[i])
- ||A|| = √(Σ(A[i]²)) = magnitude of A
- ||B|| = √(Σ(B[i]²)) = magnitude of B

Result range: [0, 1]
- 1.0 = identical preferences
- 0.5 = moderate similarity
- 0.0 = no similarity
```

## PySpark Implementation Highlights

### User-Based CF
```python
# Self-join to find user pairs with common items
user_pairs = (user_vectors.alias("u1")
              .join(user_vectors.alias("u2"), 
                    F.col("u1.category") == F.col("u2.category"))
              .where(F.col("u1.customer_id") < F.col("u2.customer_id")))

# Calculate cosine similarity
user_similarity = (user_pairs
    .groupBy("user1", "user2")
    .agg((F.sum(F.col("rating1") * F.col("rating2")) / 
          (F.sqrt(F.sum(F.col("rating1")**2)) * 
           F.sqrt(F.sum(F.col("rating2")**2))))
         .alias("similarity")))
```

### Item-Based CF
```python
# Self-join to find item pairs with common users
item_pairs = (item_vectors.alias("i1")
              .join(item_vectors.alias("i2"),
                    F.col("i1.customer_id") == F.col("i2.customer_id"))
              .where(F.col("i1.category") < F.col("i2.category")))

# Calculate cosine similarity
item_similarity = (item_pairs
    .groupBy("item1", "item2")
    .agg((F.sum(F.col("rating1") * F.col("rating2")) / 
          (F.sqrt(F.sum(F.col("rating1")**2)) * 
           F.sqrt(F.sum(F.col("rating2")**2))))
         .alias("similarity")))
```
