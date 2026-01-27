# Customer Segmentation Integration

This folder contains your production customer segmentation script integrated into the Streamlit demo.

## What's Included

### Data File
- `square_transactions.csv` - 5,000 synthetic Square POS transactions with realistic patterns

### Data Structure
The CSV matches Square's export format:
```csv
Date,Time,Item,Qty,Price Point Name,Gross Sales,Discounts,Net Sales,Tax,Transaction ID,Payment ID,Customer ID,Customer Name...
```

## Your Script Integration

The `analyses/customer_segmentation.py` module now uses your actual production code:

### Key Features Integrated

✅ **Transaction Preprocessing** - Your exact item categorization logic
✅ **Cyclical Encoding** - Sin/cos encoding for months (handles seasonality properly)
✅ **PCA Dimensionality Reduction** - Reduces to 7 principal components
✅ **DBSCAN Clustering** - Density-based clustering (eps=1.1, min_samples=50)
✅ **UMAP 3D Visualization** - Interactive 3D plot showing customer clusters
✅ **Business Hours Filtering** - Keeps only 8 AM - 5 PM transactions

### Product Categories

Your categories are preserved:
- **Bakery**: Churro, Pan Dulce, Cookies, Croissant, Muffin, Concha, Cinnamon Roll
- **Kitchen**: BLT, Avocado Toast, Breakfast Tacos, Chilaquiles, Quesadilla, Crepe  
- **Coffee**: Latte, Cappuccino, Americano, Cold Brew, Cortado, Drip
- **Other Drinks**: Horchata, Limonada, Matcha Latte, Chai, Tea

## Visualizations in the Demo

The Streamlit app shows a **4-panel dashboard**:

1. **3D UMAP Customer Segmentation** (top-left, spans 2 rows)
   - Interactive 3D scatter plot
   - Each point = one transaction
   - Color-coded by cluster
   - Rotate, zoom, and explore

2. **PCA Explained Variance** (top-right)
   - Bar chart showing how much variance each PC explains
   - Validates dimensionality reduction

3. **Cluster Size Distribution** (bottom-right)  
   - Shows number of transactions in each cluster
   - Highlights outliers in red

4. **Segment Behavioral Profile** (bottom-left)
   - Stacked bars showing average category preferences per cluster
   - Compares Bakery, Kitchen, Coffee, Other Drinks

## Technical Details

### Feature Engineering
Your exact approach:
```python
# Cyclical month encoding (handles Dec→Jan properly)
Month_sin = sin(2π × Month / 12)
Month_cos = cos(2π × Month / 12)

# Features used for clustering:
['Hour', 'Day_num', 'Bakery', 'Kitchen', 'Coffee', 
 'Other Drinks', 'Month_sin', 'Month_cos']
```

### Why DBSCAN?
- Doesn't require pre-specifying number of clusters
- Identifies outliers automatically (noise = -1)
- Works well with the spherical distribution from cyclical encoding
- Your eps=1.1 and min_samples=50 parameters are preserved

### Why UMAP?
- Better than t-SNE for preserving global structure
- 3D visualization is intuitive for presentations
- Shows natural clustering without forcing separation

## Using Your Own Data

To use actual Square exports:

1. Export transaction data from Square Dashboard
2. Save as `data/square_transactions.csv`
3. Make sure columns match the expected format
4. Run the app!

The script handles:
- Dollar sign removal from currency
- Date/time parsing  
- Missing customer IDs
- Various time formats

## Performance Notes

- **5,000 transactions**: ~3-5 seconds
- **20,000 transactions**: ~10-15 seconds
- **100,000+ transactions**: May take 1-2 minutes for UMAP

For very large datasets, consider:
- Running analysis offline and caching results
- Using a random sample for the demo
- Pre-computing UMAP embeddings

## Advanced: Customizing Parameters

In `analyses/customer_segmentation.py`, you can adjust:

```python
# DBSCAN clustering (line ~140)
db = DBSCAN(eps=1.1, min_samples=50).fit(X_scaled_df)
# eps: larger = bigger clusters, smaller = more clusters
# min_samples: higher = fewer clusters, lower = more clusters

# UMAP visualization (line ~165)
umap_reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15)
# n_neighbors: 5-15 = local structure, 50-100 = global structure
```

## What's NOT Included (From Your Original Script)

These features from your notebook aren't in the demo but could be added:

- ❌ Silhouette score optimization loop
- ❌ t-SNE visualization (UMAP is better for 3D)
- ❌ Detailed PCA loadings interpretation
- ❌ K-means comparison (DBSCAN is more appropriate here)

To add these, modify the `create_visualization()` or `perform_clustering()` methods.

## Dependencies

New requirement:
- `umap-learn>=0.5.3` - For 3D dimensionality reduction

If UMAP is not installed, the app will fall back to showing PCA projections instead (less visually appealing but still works).

## Color Scheme

The visualizations use:
- **Viridis colormap** - for continuous cluster labels
- **Purple (#6366f1)** - main analysis color (matches your branding)
- **Red (#ef4444)** - for outliers/noise points

## Next Steps

Want to enhance the demo? Consider adding:
1. **Cluster profiles table** - Show average purchase time, favorite categories per cluster
2. **Customer journey** - Track how individuals move between clusters over time
3. **Segment naming** - Automatically name clusters ("Morning Coffee Crowd", "Lunch Rush", etc.)
4. **RFM overlay** - Show how RFM segments map to behavioral clusters
