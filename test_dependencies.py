#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed correctly.
Run this before starting the Streamlit app.
"""

import sys

print("=" * 60)
print("ANALYTICS DEMO - DEPENDENCY CHECK")
print("=" * 60)

all_good = True

# Test Python version
print(f"\n1. Python Version: {sys.version.split()[0]}")
if sys.version_info < (3, 8):
    print("   ✗ Python 3.8+ required")
    all_good = False
else:
    print("   ✓ OK")

# Test Streamlit
print("\n2. Streamlit")
try:
    import streamlit
    print(f"   ✓ Installed (version {streamlit.__version__})")
except ImportError:
    print("   ✗ NOT installed - Run: pip install streamlit")
    all_good = False

# Test Plotly
print("\n3. Plotly")
try:
    import plotly
    print(f"   ✓ Installed (version {plotly.__version__})")
except ImportError:
    print("   ✗ NOT installed - Run: pip install plotly")
    all_good = False

# Test Pandas
print("\n4. Pandas")
try:
    import pandas
    print(f"   ✓ Installed (version {pandas.__version__})")
except ImportError:
    print("   ✗ NOT installed - Run: pip install pandas")
    all_good = False

# Test NumPy
print("\n5. NumPy")
try:
    import numpy
    print(f"   ✓ Installed (version {numpy.__version__})")
except ImportError:
    print("   ✗ NOT installed - Run: pip install numpy")
    all_good = False

# Test NLTK
print("\n6. NLTK (for Sentiment Analysis)")
try:
    import nltk
    print(f"   ✓ Installed (version {nltk.__version__})")
    
    # Check for VADER lexicon
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        # Test it
        test_score = sia.polarity_scores("This is great!")
        print("   ✓ VADER lexicon downloaded and working")
    except LookupError:
        print("   ⚠ VADER lexicon NOT downloaded")
        print("     Run: python -c \"import nltk; nltk.download('vader_lexicon')\"")
        all_good = False
    except Exception as e:
        print(f"   ⚠ Error testing VADER: {e}")
        all_good = False
        
except ImportError:
    print("   ✗ NOT installed - Run: pip install nltk")
    all_good = False

# Test scikit-learn
print("\n7. Scikit-learn (for Topic Extraction)")
try:
    import sklearn
    print(f"   ✓ Installed (version {sklearn.__version__})")
    
    # Test CountVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    print("   ✓ CountVectorizer working")
except ImportError:
    print("   ✗ NOT installed - Run: pip install scikit-learn")
    all_good = False
except Exception as e:
    print(f"   ⚠ Error: {e}")
    all_good = False

# Test UMAP
print("\n8. UMAP (for 3D Customer Segmentation)")
try:
    import umap
    print(f"   ✓ Installed (version {umap.__version__})")
    print("   ✓ 3D customer segmentation visualization available")
except ImportError:
    print("   ⚠ NOT installed - Run: pip install umap-learn")
    print("   ℹ Customer segmentation will fall back to PCA visualization")
    # Don't fail for UMAP - it's optional
except Exception as e:
    print(f"   ⚠ Error: {e}")

# Test data files
print("\n9. Data Files")
from pathlib import Path

data_dir = Path("data")
if not data_dir.exists():
    print("   ✗ data/ directory not found")
    all_good = False
else:
    print("   ✓ data/ directory exists")
    
    # Check for Google reviews JSON
    json_file = data_dir / "google_reviews.json"
    if json_file.exists():
        print("   ✓ google_reviews.json found")
        
        # Try to load it
        try:
            import json
            with open(json_file) as f:
                data = json.load(f)
                if 'reviews' in data:
                    print(f"   ✓ Valid JSON with {len(data['reviews'])} reviews")
                else:
                    print("   ⚠ JSON missing 'reviews' key")
        except Exception as e:
            print(f"   ⚠ Error reading JSON: {e}")
    else:
        print("   ⚠ google_reviews.json not found (Sentiment Analysis will not work)")

# Test analyses modules
print("\n10. Analysis Modules")
try:
    from analyses import (
        SentimentAnalysis,
        TopicExtraction,
        ComputerVision,
        DemandForecasting,
        EDADescriptive,
        SeasonalityTimeSeries,
        CustomerSegmentation,
        BasketAnalysis,
        ChurnModeling
    )
    print("   ✓ All 9 analysis modules imported successfully")
except ImportError as e:
    print(f"   ✗ Error importing modules: {e}")
    all_good = False

# Final result
print("\n" + "=" * 60)
if all_good:
    print("✓ ALL CHECKS PASSED!")
    print("\nYou're ready to run the app:")
    print("  streamlit run analytics_showcase_refactored.py")
else:
    print("✗ SOME CHECKS FAILED")
    print("\nPlease install missing dependencies:")
    print("  pip install -r requirements.txt")
    print("  python -c \"import nltk; nltk.download('vader_lexicon')\"")
print("=" * 60)

sys.exit(0 if all_good else 1)
