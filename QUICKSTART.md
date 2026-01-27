# Quick Start Guide - Ron's HVAC Case Study

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NLTK data (for sentiment analysis)
python -c "import nltk; nltk.download('vader_lexicon')"

# 3. Run the app
streamlit run app.py
```

That's it! The app will open in your browser.

## 📖 What You'll See

### 1. Introduction Page
- Ron's story and challenges
- Overview of the 9 analyses
- "Start the Analysis Journey" button

### 2. Sequential Analysis Journey
The analyses tell a story in this order:

1. **Business Overview** - Ron's current state (revenue, customers, services)
2. **Customer Segmentation** - Who are his customers?
3. **Sentiment Analysis** - What do they say about him?
4. **Topic Extraction** - What specific themes come up?
5. **Churn Prediction** - Who's at risk of leaving?
6. **Demand Forecasting** - When will he be busy?
7. **Seasonality Analysis** - What's normal vs concerning?
8. **Market Basket** - Service bundling opportunities

### 3. Final Synthesis
- How everything connects
- Top 5 findings
- Prioritized 90-day action plan
- Projected impact

## 🎯 Navigation

- **Sidebar**: Jump to any analysis
- **Top buttons**: Navigate through 5 steps per analysis
- **Bottom buttons**: Previous/Next
- **Progress bar**: Shows where you are (Step X of 9)

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'nltk'"
**Solution**: Run `pip install nltk scikit-learn`

### Error: "Resource vader_lexicon not found"
**Solution**: Run `python -c "import nltk; nltk.download('vader_lexicon')"`

### Error: "No module named 'umap'"
**Solution**: Run `pip install umap-learn` (optional - for 3D customer segmentation)

### Error: Can't find data files
**Solution**: Make sure you're running from the `analytics_demo` directory

## 📁 File Structure

```
analytics_demo/
├── app.py                    # Main app - RUN THIS
├── intro_page.py             # Introduction
├── synthesis_page.py         # Final synthesis
├── analyses/
│   ├── business_overview.py  # NEW - Analysis #1
│   ├── customer_segmentation.py
│   ├── sentiment_analysis.py
│   └── ... (8 total)
└── data/
    └── ... (CSV and JSON files)
```

## 💡 Tips for Demoing

1. **Start with the intro** - Sets up the narrative
2. **Walk through Business Overview** - Establishes Ron's challenges
3. **Show 2-3 middle analyses** - Demonstrate how they build on each other
4. **Jump to Synthesis** - Show the complete picture

Don't feel like you need to click through all 5 steps of every analysis - the sidebar lets you jump around!

## 🎨 What Makes This Different

This isn't just a collection of analyses - it's a **complete business case study**:

✅ **Narrative structure** - Each analysis builds on the previous
✅ **Ron-specific insights** - Not generic templates
✅ **Cross-references** - "Remember from Analysis #2..."
✅ **Action-oriented** - Every insight leads to a recommendation
✅ **Client-friendly** - Respects that Ron is a business owner, not a data scientist

## 🔄 Making Changes

Want to customize for a different client?

1. Update `intro_page.py` - Change Ron's story
2. Update each analysis in `analyses/` - Change insights/recommendations
3. Update `synthesis_page.py` - Change the action plan

The structure stays the same, just swap the content!
