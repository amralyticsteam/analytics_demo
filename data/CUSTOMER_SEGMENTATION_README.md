# Customer Segmentation Transaction Dataset

## Overview
This synthetic dataset contains 10,000 beauty retail transactions from 1,840 unique customers over a 2-year period (2022-2023). It's designed for customer segmentation analysis using clustering algorithms.

## Dataset Characteristics
- **Rows**: 10,000 transactions
- **Unique Customers**: 1,840
- **Date Range**: January 1, 2022 - December 31, 2023
- **Industry**: Beauty Retail
- **Average Transaction Value**: $81.82
- **Average Categories per Transaction**: 2.1

## Column Descriptions

### Transaction Identifiers
- `transaction_id`: Unique identifier for each transaction (1-10000)
- `customer_id`: Unique customer identifier (1-2000)
- `transaction_date`: Date of purchase (YYYY-MM-DD format)

### Time-Based Features
- `day_of_week`: Day of the week (Monday-Sunday)
- `hour`: Hour of day (9-20, representing 9 AM - 8 PM)
- `season`: Season of purchase (Winter, Spring, Summer, Fall)

### Product Categories (One-Hot Encoded)
Each column indicates whether the transaction included items from that category (1 = yes, 0 = no):
- `has_Skincare`: Skincare products (cleansers, moisturizers, serums, etc.)
- `has_Haircare`: Haircare products (shampoo, conditioner, styling products, etc.)
- `has_Cosmetics`: Makeup products (foundation, lipstick, eyeshadow, etc.)
- `has_Fragrance`: Perfumes and colognes
- `has_Bath_Body`: Bath and body products (lotions, body wash, etc.)
- `has_Tools_Accessories`: Beauty tools (brushes, sponges, mirrors, etc.)
- `has_Gifts`: Gift sets and special gift items
- `has_Wellness`: Wellness products (supplements, aromatherapy, etc.)

### Purchase Details
- `num_categories`: Total number of different product categories in the transaction (1-7)
- `subtotal`: Purchase amount before discount ($)
- `discount_percent`: Discount percentage applied (0-30%)
- `discount_amount`: Dollar amount of discount ($)
- `total_amount`: Final purchase amount after discount ($)

### Customer Attributes
These attributes are consistent for each customer across all their transactions:
- `loyalty_member`: Customer is enrolled in loyalty program (1 = yes, 0 = no)
- `newsletter_subscriber`: Customer subscribed to email newsletter (1 = yes, 0 = no)
- `sms_subscriber`: Customer subscribed to SMS/text updates (1 = yes, 0 = no)

## Category Purchase Rates
- Skincare: 53.3% (most popular)
- Cosmetics: 40.2%
- Haircare: 29.9%
- Bath & Body: 25.4%
- Fragrance: 19.8%
- Gifts: 15.5% (seasonal spikes in Nov, Dec, Feb, May)
- Tools & Accessories: 15.2%
- Wellness: 11.8%

## Data Patterns & Insights

### Customer Behavior
- Some customers are frequent purchasers (multiple transactions)
- Transaction frequency follows a power law distribution (few heavy buyers, many light buyers)
- Loyalty members receive discounts more frequently (60% of discount transactions)

### Seasonal Patterns
- Gift purchases spike during holidays (November, December, February, May)
- Seasonal sales increase discount rates in November, December, and July
- Q4 (October-December) typically shows higher transaction volumes

### Pricing Structure
- Average category base prices:
  - Fragrance: ~$85 (highest)
  - Gifts: ~$55
  - Skincare: ~$45
  - Tools & Accessories: ~$38
  - Wellness: ~$35
  - Cosmetics: ~$32
  - Haircare: ~$28
  - Bath & Body: ~$22 (lowest)

### Discount Patterns
- Loyalty members: 40% receive discounts vs 20% for non-members
- Discount ranges: 0%, 10%, 15%, 20%, or 30%
- Seasonal boost discounts in July and November/December

## Use Cases for Segmentation

This dataset is ideal for:

1. **RFM Analysis** (Recency, Frequency, Monetary)
   - Recency: Days since last purchase
   - Frequency: Number of transactions per customer
   - Monetary: Total spending per customer

2. **Customer Clustering**
   - K-means clustering on purchase behavior
   - Hierarchical clustering for segment discovery
   - DBSCAN for outlier detection

3. **Product Affinity Analysis**
   - Which categories are purchased together?
   - Cross-sell opportunities
   - Bundle recommendations

4. **Lifetime Value Prediction**
   - Predict future customer value
   - Identify high-value segments
   - Optimize marketing spend

5. **Churn Prediction**
   - Identify customers with declining activity
   - Flag at-risk customers
   - Proactive retention campaigns

## Suggested Segmentation Approaches

### Feature Engineering Ideas
```python
# Customer-level aggregations
customer_features = df.groupby('customer_id').agg({
    'transaction_id': 'count',  # Purchase frequency
    'total_amount': ['sum', 'mean', 'std'],  # Spending patterns
    'discount_percent': 'mean',  # Discount sensitivity
    'transaction_date': ['min', 'max'],  # Recency calculation
    'has_Skincare': 'mean',  # Category preferences
    'has_Cosmetics': 'mean',
    # ... repeat for each category
})

# Calculate recency
from datetime import datetime
max_date = df['transaction_date'].max()
customer_features['days_since_last_purchase'] = (
    max_date - df.groupby('customer_id')['transaction_date'].max()
).dt.days
```

### Potential Segments to Look For
- **VIP Champions**: High frequency, high spend, recent purchases
- **Loyal Regulars**: Consistent frequency, moderate spend
- **Bargain Hunters**: High discount sensitivity, low full-price purchases
- **Category Specialists**: Strong preference for 1-2 categories
- **Multi-Category Explorers**: Purchase across many categories
- **At-Risk**: Declining frequency, increasing time between purchases
- **One-Time Buyers**: Single transaction, no return

## Data Quality Notes

- No missing values
- All transactions have at least one product category
- Customer attributes (loyalty_member, newsletter_subscriber, sms_subscriber) are consistent per customer
- Realistic price variation and discount patterns
- Date range spans 2 full years for seasonality analysis

## Loading the Data

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('customer_segmentation_transactions.csv')

# Convert date column to datetime
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# Basic exploration
print(f"Shape: {df.shape}")
print(f"Customers: {df['customer_id'].nunique()}")
print(f"Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
```

## Next Steps

1. Aggregate transaction-level data to customer-level features
2. Perform exploratory data analysis (EDA)
3. Engineer features for segmentation
4. Apply clustering algorithms (K-means, hierarchical, DBSCAN)
5. Evaluate and interpret segments
6. Create actionable business recommendations for each segment
