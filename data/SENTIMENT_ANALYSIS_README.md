# Sentiment Analysis Integration

This folder contains your production sentiment analysis script integrated into the Streamlit demo.

## What's Included

### Data File
- `google_reviews.json` - 200 synthetic Google reviews in the exact format your script expects

### JSON Structure
```json
{
  "reviews": [
    {
      "reviewer": {
        "displayName": "John D."
      },
      "starRating": "FIVE",
      "comment": "Amazing service and quality!",
      "createTime": "2024-06-13T00:00:00.000Z",
      "name": "accounts/12345/locations/67890/reviews/review_0001"
    }
  ]
}
```

## Your Script Integration

The `analyses/sentiment_analysis.py` module now uses your actual production code:

1. **JSON Ingestion** - Uses your `ingest_json_data()` function
2. **VADER Sentiment** - Uses your `analyze_sentiment()` with VADER
3. **Topic Extraction** - Uses your `extract_topics()` with CountVectorizer
4. **Custom Colors** - Uses your teal color palette (#008f8c, #023535, etc.)

## Key Features Integrated

✅ Maps star ratings (FIVE, FOUR, etc.) to numeric values  
✅ VADER sentiment scoring (-1 to +1)  
✅ Sentiment labels (Positive/Neutral/Negative)  
✅ Topic extraction for 2-3 word phrases  
✅ Removes "Translated by Google" from text  
✅ Handles date parsing and cleaning  

## Visualization

The Streamlit visualization shows:
- Sentiment distribution pie chart
- Average sentiment by star rating
- Top positive topics (horizontal bar chart)
- Top negative topics (horizontal bar chart)

All using your custom teal color scheme!

## Using Your Own Data

To use your actual client data:

1. Export Google reviews to JSON format
2. Place the JSON file in the `data/` folder
3. Update `data_file` property in `sentiment_analysis.py`:
   ```python
   @property
   def data_file(self) -> str:
       return 'your_actual_reviews.json'
   ```

## Adding More Features

Your original script includes features not yet in the demo:

- **Word Clouds** - Can be added to `create_visualization()`
- **Excel Reports** - Can be exported as downloadable files
- **Sample Reviews Display** - Can be shown in the Insights section

To add these, modify the `create_visualization()` method in `analyses/sentiment_analysis.py`.

## Dependencies

The script requires:
- nltk (for VADER sentiment analysis)
- scikit-learn (for CountVectorizer topic extraction)

These are now in `requirements.txt` and will be installed automatically.

## Color Palette Reference

Your custom colors (from the original script):
- **Dark Teal**: #23606e
- **Cream**: #fff8de  
- **Very Dark Teal**: #023535
- **Medium Teal**: #008f8c (used for positive)
- **Standard Teal**: #015958

The demo uses #008f8c for positive and #023535 for negative, matching your original design.
