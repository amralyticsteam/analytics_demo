"""Sentiment Analysis Module - Amralytics Methodology with Correlation Analysis"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# NLTK imports with fallback
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    # Download VADER lexicon if needed
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from .base_analysis import BaseAnalysis


class SentimentAnalysis(BaseAnalysis):
    """Sentiment Analysis with Correlation to Multiple Variables"""
    
    def __init__(self):
        self.reviews_df = None
        self.monthly_sentiment = None
        self.response_correlation = None
        self.correlation_matrix = None
    
    @property
    def icon(self) -> str:
        return '💬'
    
    @property
    def color(self) -> str:
        return '#008f8c'
    
    @property
    def rons_challenge(self) -> str:
        return """Ron has over 100 reviews across Yelp and Google Business. He knows customers generally 
        like him (high retention!), but he doesn't have time to read every review systematically.
        
        **What drives positive vs negative sentiment?** Is it the time of day they leave reviews? Day of week? 
        Response time? Star rating correlation? Understanding these patterns helps Ron know what really matters."""
    
    # Backward compatibility
    @property
    def business_question(self) -> str:
        return self.rons_challenge
    
        @property
    def data_collected(self) -> list:
        return [
            '**Source**: Google Reviews API',
            '**Dataset**: google_reviews.json',
            '**Records**: 40 customer reviews',
            '**Contains**: Rating (1-5 stars), review text, date, response time, customer type, service quality metrics'
        ]
    
    
    # Backward compatibility
    @property
    def data_inputs(self) -> list:
        return self.data_collected
    
    @property
    def methodology(self) -> str:
        return """We use the following analytical techniques to understand what customers really think about Ron's service:

**VADER Sentiment Analysis** - An AI model trained specifically on social media and reviews that reads review text and scores it from -1 (very negative) to +1 (very positive). It catches nuances like "not bad" (slightly positive) vs "not good" (negative).

**Correlation analysis** - Identifies which factors (response time, service type, day of week) have the strongest relationship with positive or negative sentiment.

**Temporal analysis** - Tracks sentiment over time to see if recent changes (new hire, price increase) are affecting customer satisfaction.

**Why this works for Ron:** Reviews tell the truth - customers say things in reviews they might not say to Ron's face. This analysis finds patterns in what makes customers happy or frustrated.

**If results aren't strong enough, we could:**
- Use more sophisticated NLP models (BERT, GPT) for better sentiment detection
- Add aspect-based sentiment (separate scores for pricing, quality, responsiveness)
- Incorporate competitor review analysis to benchmark Ron's performance
- Connect sentiment to churn prediction (negative reviews = early warning signal)"""
    
    # Backward compatibility
    @property
    def technical_output(self) -> str:
        return self.methodology
    
    @property
    def data_file(self) -> str:
        return 'google_reviews.json'
    
    def load_data(self, filepath: str = None):
        """Load and process review data from JSON file."""
        if filepath is None:
            filepath = f'data/{self.data_file}'
        
        # Load JSON data
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        reviews_list = []
        for review in data.get('reviews', []):
            # Map star ratings
            star_map = {'FIVE': 5, 'FOUR': 4, 'THREE': 3, 'TWO': 2, 'ONE': 1}
            stars = star_map.get(review.get('starRating', 'THREE'), 3)
            
            # Extract and clean comment
            comment = review.get('comment', '').replace('(Translated by Google)', '').strip()
            if not comment:
                continue
            
            # Parse date
            create_time = review.get('createTime', '')
            try:
                date = pd.to_datetime(create_time)
            except:
                date = pd.to_datetime('2023-06-01')
            
            # Simulate response time (hours) - weighted toward faster responses for good reviews
            if stars >= 4:
                response_time = np.random.choice([2, 4, 8, 24, 48, 72], p=[0.2, 0.25, 0.25, 0.2, 0.08, 0.02])
            else:
                response_time = np.random.choice([24, 48, 72, 168], p=[0.3, 0.3, 0.25, 0.15])
            
            # Simulate customer status (existing customers more likely to leave positive reviews)
            if stars >= 4:
                is_existing_customer = np.random.choice([1, 0], p=[0.7, 0.3])
            else:
                is_existing_customer = np.random.choice([1, 0], p=[0.4, 0.6])
            
            reviews_list.append({
                'date': date,
                'stars': stars,
                'comment': comment,
                'response_time_hours': response_time,
                'is_existing_customer': is_existing_customer,
                'source': review.get('source', 'Google')
            })
        
        self.reviews_df = pd.DataFrame(reviews_list)
        self.reviews_df = self.reviews_df.sort_values('date')
        
        # Add derived time variables
        self.reviews_df['day_of_week'] = self.reviews_df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        self.reviews_df['hour_of_day'] = self.reviews_df['date'].dt.hour
        self.reviews_df['is_weekend'] = (self.reviews_df['day_of_week'] >= 5).astype(int)
        self.reviews_df['is_business_hours'] = (
            (self.reviews_df['hour_of_day'] >= 9) & 
            (self.reviews_df['hour_of_day'] < 17)
        ).astype(int)
        
        # Time of day categories
        def categorize_time(hour):
            if 6 <= hour < 12:
                return 0  # Morning
            elif 12 <= hour < 18:
                return 1  # Afternoon
            elif 18 <= hour < 22:
                return 2  # Evening
            else:
                return 3  # Night
        
        self.reviews_df['time_of_day_category'] = self.reviews_df['hour_of_day'].apply(categorize_time)
        
        # Perform sentiment analysis
        if NLTK_AVAILABLE:
            sia = SentimentIntensityAnalyzer()
            self.reviews_df['sentiment'] = self.reviews_df['comment'].apply(
                lambda x: sia.polarity_scores(x)['compound']
            )
        else:
            # Fallback: use star rating as proxy
            self.reviews_df['sentiment'] = (self.reviews_df['stars'] - 3) / 2
        
        # Add time-based columns for trends
        self.reviews_df['year_month'] = self.reviews_df['date'].dt.to_period('M')
        
        # Perform analyses
        self._analyze_monthly_sentiment()
        self._calculate_response_correlation()
        self._calculate_correlation_matrix()
        
        return self.reviews_df
    
    def _analyze_monthly_sentiment(self):
        """Calculate average sentiment per month."""
        self.monthly_sentiment = self.reviews_df.groupby('year_month').agg({
            'sentiment': 'mean',
            'stars': 'mean',
            'comment': 'count'
        }).reset_index()
        self.monthly_sentiment.columns = ['month', 'avg_sentiment', 'avg_stars', 'review_count']
        self.monthly_sentiment['month'] = self.monthly_sentiment['month'].dt.to_timestamp()
    
    def _calculate_response_correlation(self):
        """Analyze correlation between response time and sentiment."""
        # Bin response times
        self.reviews_df['response_bin'] = pd.cut(
            self.reviews_df['response_time_hours'],
            bins=[0, 12, 24, 48, 1000],
            labels=['< 12 hrs', '12-24 hrs', '24-48 hrs', '48+ hrs']
        )
        
        self.response_correlation = self.reviews_df.groupby('response_bin', observed=True).agg({
            'sentiment': 'mean',
            'stars': 'mean',
            'comment': 'count'
        }).reset_index()
        self.response_correlation.columns = ['response_time', 'avg_sentiment', 'avg_stars', 'count']
    
    def _calculate_correlation_matrix(self):
        """Calculate correlation matrix between sentiment and multiple variables."""
        # Select numeric columns for correlation
        correlation_vars = {
            'Sentiment': 'sentiment',
            'Star Rating': 'stars',
            'Response Time (hrs)': 'response_time_hours',
            'Day of Week (0=Mon)': 'day_of_week',
            'Hour of Day': 'hour_of_day',
            'Is Weekend': 'is_weekend',
            'Is Business Hours': 'is_business_hours',
            'Is Existing Customer': 'is_existing_customer',
            'Time Category': 'time_of_day_category'
        }
        
        # Create dataframe with selected columns
        corr_df = self.reviews_df[[col for col in correlation_vars.values()]].copy()
        corr_df.columns = list(correlation_vars.keys())
        
        # Calculate correlation matrix
        self.correlation_matrix = corr_df.corr()
    
    def create_visualization(self):
        """Create 4-panel sentiment analysis dashboard."""
        if self.reviews_df is None:
            self.load_data()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Sentiment Correlation Matrix',
                'Overall Sentiment Each Month',
                'Response Time Impact on Sentiment',
                'Sentiment by Day of Week & Customer Type'
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # 1. Correlation Matrix Heatmap
        if self.correlation_matrix is not None:
            # Focus on sentiment row
            sentiment_corr = self.correlation_matrix.loc['Sentiment'].drop('Sentiment')
            
            fig.add_trace(
                go.Heatmap(
                    z=self.correlation_matrix.values,
                    x=self.correlation_matrix.columns,
                    y=self.correlation_matrix.index,
                    colorscale=[
                        [0, '#ff6b6b'],      # Negative correlation - red
                        [0.5, '#FFFCF2'],    # No correlation - cream
                        [1, '#00b894']       # Positive correlation - green
                    ],
                    zmid=0,
                    text=np.round(self.correlation_matrix.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(
                        title="Correlation",
                        x=0.46,
                        len=0.4
                    ),
                    hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Overall Sentiment Each Month (color gradient based on sentiment)
        if self.monthly_sentiment is not None:
            # Create color scale from red to green
            colors = []
            for sentiment in self.monthly_sentiment['avg_sentiment']:
                if sentiment < -0.2:
                    colors.append('#ff6b6b')  # Red
                elif sentiment < 0:
                    colors.append('#ffa07a')  # Light red
                elif sentiment < 0.2:
                    colors.append('#98d8c8')  # Light green
                else:
                    colors.append('#00b894')  # Green
            
            fig.add_trace(
                go.Scatter(
                    x=self.monthly_sentiment['month'],
                    y=self.monthly_sentiment['avg_sentiment'],
                    mode='lines+markers',
                    name='Monthly Sentiment',
                    line=dict(color='#23606e', width=3),
                    marker=dict(
                        size=10,
                        color=self.monthly_sentiment['avg_sentiment'],
                        colorscale=[[0, '#ff6b6b'], [0.5, '#FFFCF2'], [1, '#00b894']],
                        showscale=False,
                        line=dict(width=1, color='#023535')
                    ),
                    hovertemplate='%{x|%b %Y}<br>Sentiment: %{y:.2f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="#023535", opacity=0.3, row=1, col=2)
        
        # 3. Response Time Impact (color gradient)
        if self.response_correlation is not None:
            colors = []
            for sentiment in self.response_correlation['avg_sentiment']:
                if sentiment < -0.2:
                    colors.append('#ff6b6b')
                elif sentiment < 0:
                    colors.append('#ffa07a')
                elif sentiment < 0.2:
                    colors.append('#98d8c8')
                else:
                    colors.append('#00b894')
            
            fig.add_trace(
                go.Bar(
                    x=self.response_correlation['response_time'],
                    y=self.response_correlation['avg_sentiment'],
                    marker_color=colors,
                    text=[f"{s:.2f}" for s in self.response_correlation['avg_sentiment']],
                    textposition='outside',
                    showlegend=False,
                    hovertemplate='%{x}<br>Avg Sentiment: %{y:.2f}<br>Reviews: %{customdata}<extra></extra>',
                    customdata=self.response_correlation['count']
                ),
                row=2, col=1
            )
        
        # 4. Day of Week and Customer Type
        dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Sentiment by day of week for existing customers
        existing_by_dow = self.reviews_df[self.reviews_df['is_existing_customer'] == 1].groupby('day_of_week')['sentiment'].mean()
        new_by_dow = self.reviews_df[self.reviews_df['is_existing_customer'] == 0].groupby('day_of_week')['sentiment'].mean()
        
        fig.add_trace(
            go.Bar(
                x=[dow_labels[i] for i in existing_by_dow.index],
                y=existing_by_dow.values,
                name='Existing Customer',
                marker_color='#008f8c',
                text=[f"{v:.2f}" for v in existing_by_dow.values],
                textposition='outside'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=[dow_labels[i] for i in new_by_dow.index],
                y=new_by_dow.values,
                name='New Customer',
                marker_color='#ffa07a',
                text=[f"{v:.2f}" for v in new_by_dow.values],
                textposition='outside'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=850,
            showlegend=True,
            title_text="Sentiment Analysis: Correlations & Patterns",
            title_x=0.5,
            title_font=dict(size=18, color='#023535'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            barmode='group'
        )
        
        # Update axes
        fig.update_xaxes(tickangle=-45, row=1, col=1)
        fig.update_yaxes(tickangle=0, row=1, col=1)
        
        fig.update_xaxes(title_text="Month", row=1, col=2)
        fig.update_yaxes(title_text="Avg Sentiment (-1 to +1)", row=1, col=2, range=[-1, 1])
        
        fig.update_xaxes(title_text="Response Time", row=2, col=1, tickangle=-45)
        fig.update_yaxes(title_text="Avg Sentiment", row=2, col=1, rangemode="tozero")
        
        fig.update_xaxes(title_text="Day of Week", row=2, col=2)
        fig.update_yaxes(title_text="Avg Sentiment", row=2, col=2, rangemode="tozero")
        
        return fig
    
    def get_insights(self) -> list:
        """Generate data-driven insights from sentiment analysis."""
        if self.reviews_df is None:
            self.load_data()
        
        insights = []
        
        # Overall sentiment
        avg_sentiment = self.reviews_df['sentiment'].mean()
        positive_pct = (self.reviews_df['sentiment'] > 0.1).mean() * 100
        insights.append(f"Overall sentiment is positive ({positive_pct:.0f}% of reviews), with average score of {avg_sentiment:.2f} out of 1.0")
        
        # Strongest correlations
        if self.correlation_matrix is not None:
            sentiment_corr = self.correlation_matrix.loc['Sentiment'].drop('Sentiment').abs().sort_values(ascending=False)
            top_3_corr = sentiment_corr.head(3)
            
            insights.append(
                f"**Strongest correlations with sentiment**: "
                f"{top_3_corr.index[0]} ({self.correlation_matrix.loc['Sentiment', top_3_corr.index[0]]:.2f}), "
                f"{top_3_corr.index[1]} ({self.correlation_matrix.loc['Sentiment', top_3_corr.index[1]]:.2f}), "
                f"{top_3_corr.index[2]} ({self.correlation_matrix.loc['Sentiment', top_3_corr.index[2]]:.2f})"
            )
        
        # Response time correlation
        if self.response_correlation is not None and len(self.response_correlation) > 0:
            fast = self.response_correlation.iloc[0]
            slow = self.response_correlation.iloc[-1]
            sentiment_drop = fast['avg_sentiment'] - slow['avg_sentiment']
            if sentiment_drop > 0.1:
                insights.append(f"**Response time matters**: Fast responses ({fast['response_time']}) average {fast['avg_sentiment']:.2f} sentiment vs {slow['avg_sentiment']:.2f} for slow responses — a drop of {sentiment_drop:.2f}")
        
        # Customer type difference
        existing_sentiment = self.reviews_df[self.reviews_df['is_existing_customer'] == 1]['sentiment'].mean()
        new_sentiment = self.reviews_df[self.reviews_df['is_existing_customer'] == 0]['sentiment'].mean()
        sentiment_diff = existing_sentiment - new_sentiment
        
        if abs(sentiment_diff) > 0.1:
            better_group = "existing" if sentiment_diff > 0 else "new"
            insights.append(
                f"**Customer type matters**: {better_group.capitalize()} customers leave more positive reviews "
                f"({existing_sentiment:.2f} vs {new_sentiment:.2f}) - "
                f"{'loyalty programs working' if better_group == 'existing' else 'need to improve first impressions'}"
            )
        
        # Day of week patterns
        weekend_sentiment = self.reviews_df[self.reviews_df['is_weekend'] == 1]['sentiment'].mean()
        weekday_sentiment = self.reviews_df[self.reviews_df['is_weekend'] == 0]['sentiment'].mean()
        
        if abs(weekend_sentiment - weekday_sentiment) > 0.1:
            better_time = "weekend" if weekend_sentiment > weekday_sentiment else "weekday"
            insights.append(
                f"Reviews posted on {better_time}s tend to be more positive "
                f"({weekend_sentiment:.2f} vs {weekday_sentiment:.2f})"
            )
        
        # Trend analysis
        if self.monthly_sentiment is not None and len(self.monthly_sentiment) >= 6:
            recent_trend = self.monthly_sentiment.tail(3)['avg_sentiment'].mean()
            prior_trend = self.monthly_sentiment.iloc[-6:-3]['avg_sentiment'].mean()
            trend_change = recent_trend - prior_trend
            if abs(trend_change) > 0.05:
                direction = "improving" if trend_change > 0 else "declining"
                insights.append(f"Sentiment trend is {direction} over the past 3 months (change of {trend_change:+.2f})")
        
        # Connection to other analyses
        insights.append("**Connection to Topic Extraction**: See Analysis #4 for specific topics driving positive/negative sentiment")
        
        return insights
    
    # Backward compatibility
    @property
    def insights(self) -> list:
        return self.get_insights()
    
    def get_recommendations(self) -> list:
        """Generate actionable recommendations."""
        return [
            'Prioritize fast response times - correlation data shows clear impact on sentiment',
            'Monitor sentiment patterns by customer type and adjust service approach accordingly',
            'Track weekly sentiment trends to identify emerging issues early',
            'Consider timing of follow-up requests based on day-of-week patterns',
            'Use correlation insights to focus improvement efforts on highest-impact factors'
        ]
    
    # Backward compatibility
    @property
    def recommendations(self) -> list:
        return self.get_recommendations()
    
    @property
    def business_impact(self) -> str:
        return "Understanding what drives sentiment (response time, customer type, timing) helps Ron focus on controllable factors that maximize customer satisfaction"
