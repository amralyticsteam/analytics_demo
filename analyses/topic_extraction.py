"""Topic Extraction Module - Amralytics Methodology"""

import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# NLTK imports with fallback
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    from sklearn.feature_extraction.text import CountVectorizer
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


class TopicExtraction(BaseAnalysis):
    """Topic extraction with quarterly sentiment analysis and common phrases"""
    
    def __init__(self):
        self.reviews_df = None
        self.quarterly_topics = None
        self.common_phrases = None
        self.topic_sentiments = None
    
    @property
    def icon(self) -> str:
        return '🏷️'
    
    @property
    def color(self) -> str:
        return '#23606e'
    
    @property
    def rons_challenge(self) -> str:
        return """Beyond overall sentiment, Ron needs to know **what specific topics** customers talk about. 
        Are they praising his response time? Complaining about scheduling? Appreciating his professionalism?
        
**What themes appear in reviews over time?** Understanding topic trends helps Ron focus improvements 
on what matters most and double down on what customers love."""
    
    # Backward compatibility
    @property
    def business_question(self) -> str:
        return self.rons_challenge
    
    @property
    def data_collected(self) -> list:
        return [
            'Customer review text (40 reviews) - **Google Reviews**',
            'Review ratings and dates - **Google Reviews**',
            'Common phrases and keywords - **Google Reviews**',
            'Service-specific feedback - **Google Reviews**'
    ]
    
    # Backward compatibility
    @property
    def data_inputs(self) -> list:
        return self.data_collected
    
    @property
    def methodology(self) -> str:
        return 'N-gram extraction (2-3 word phrases) using CountVectorizer, VADER sentiment scoring per topic, quarterly topic analysis showing most/least positive themes, topic frequency analysis'
    
    # Backward compatibility
    @property
    def technical_output(self) -> str:
        return self.methodology
    
    @property
    def data_file(self) -> str:
        return 'google_reviews.json'
    
    def load_data(self, filepath: str = None):
        """Load and process review data for topic extraction."""
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
            
            reviews_list.append({
                'date': date,
                'stars': stars,
                'comment': comment,
                'source': review.get('source', 'Google')
            })
        
        self.reviews_df = pd.DataFrame(reviews_list)
        self.reviews_df = self.reviews_df.sort_values('date')
        
        # Perform sentiment analysis
        if NLTK_AVAILABLE:
            sia = SentimentIntensityAnalyzer()
            self.reviews_df['sentiment'] = self.reviews_df['comment'].apply(
                lambda x: sia.polarity_scores(x)['compound']
            )
        else:
            # Fallback: use star rating as proxy
            self.reviews_df['sentiment'] = (self.reviews_df['stars'] - 3) / 2
        
        # Add time-based columns
        self.reviews_df['quarter'] = self.reviews_df['date'].dt.to_period('Q')
        self.reviews_df['year_month'] = self.reviews_df['date'].dt.to_period('M')
        
        # Extract topics
        self._extract_topics()
        
        # Analyze quarterly topics
        self._analyze_quarterly_topics()
        
        # Extract most common phrases overall
        self._extract_common_phrases()
        
        return self.reviews_df
    
    def _extract_topics(self):
        """Extract topics from all reviews."""
        if not NLTK_AVAILABLE:
            return
        
        # Extract 2-3 word phrases
        vectorizer = CountVectorizer(
            ngram_range=(2, 3), 
            max_features=40, 
            stop_words='english',
            min_df=2
        )
        
        try:
            X = vectorizer.fit_transform(self.reviews_df['comment'])
            topics = vectorizer.get_feature_names_out()
            
            # Calculate sentiment for each topic
            topic_data = []
            for topic in topics:
                mask = self.reviews_df['comment'].str.contains(topic, case=False, na=False)
                if mask.sum() >= 2:
                    topic_reviews = self.reviews_df.loc[mask]
                    topic_data.append({
                        'topic': topic,
                        'sentiment': topic_reviews['sentiment'].mean(),
                        'count': mask.sum(),
                        'avg_stars': topic_reviews['stars'].mean()
                    })
            
            self.topic_sentiments = pd.DataFrame(topic_data).sort_values('sentiment')
            
        except Exception as e:
            print(f"Topic extraction warning: {e}")
            self.topic_sentiments = pd.DataFrame()
    
    def _analyze_quarterly_topics(self):
        """Find most and least positive topics per quarter for past year."""
        if not NLTK_AVAILABLE or not hasattr(self, 'topic_sentiments') or len(self.topic_sentiments) == 0:
            return
        
        # Get last 4 quarters
        recent_quarters = self.reviews_df['quarter'].unique()[-4:]
        
        quarterly_results = []
        
        for quarter in recent_quarters:
            quarter_reviews = self.reviews_df[self.reviews_df['quarter'] == quarter]
            
            # Find topics in this quarter
            quarter_topics = []
            for _, topic_row in self.topic_sentiments.iterrows():
                topic = topic_row['topic']
                mask = quarter_reviews['comment'].str.contains(topic, case=False, na=False)
                
                if mask.sum() >= 2:
                    quarter_topics.append({
                        'quarter': quarter.to_timestamp(),
                        'topic': topic,
                        'sentiment': quarter_reviews.loc[mask, 'sentiment'].mean(),
                        'mentions': mask.sum()
                    })
            
            if quarter_topics:
                quarter_df = pd.DataFrame(quarter_topics)
                
                # Get most and least positive
                if len(quarter_df) > 0:
                    most_pos = quarter_df.nlargest(1, 'sentiment').iloc[0]
                    least_pos = quarter_df.nsmallest(1, 'sentiment').iloc[0]
                    
                    quarterly_results.append({
                        'quarter': quarter.to_timestamp(),
                        'quarter_label': quarter.strftime('%Y Q%q'),
                        'most_positive_topic': most_pos['topic'],
                        'most_positive_sentiment': most_pos['sentiment'],
                        'most_positive_mentions': most_pos['mentions'],
                        'least_positive_topic': least_pos['topic'],
                        'least_positive_sentiment': least_pos['sentiment'],
                        'least_positive_mentions': least_pos['mentions']
                    })
        
        if quarterly_results:
            self.quarterly_topics = pd.DataFrame(quarterly_results)
    
    def _extract_common_phrases(self):
        """Extract most common phrases overall."""
        if not NLTK_AVAILABLE:
            return
        
        # Use broader set for common phrases (2-4 words, higher max_features)
        vectorizer = CountVectorizer(
            ngram_range=(2, 4), 
            max_features=20, 
            stop_words='english',
            min_df=3
        )
        
        try:
            X = vectorizer.fit_transform(self.reviews_df['comment'])
            phrases = vectorizer.get_feature_names_out()
            phrase_counts = X.sum(axis=0).A1
            
            phrase_data = []
            for phrase, count in zip(phrases, phrase_counts):
                # Get sentiment for this phrase
                mask = self.reviews_df['comment'].str.contains(phrase, case=False, na=False)
                if mask.sum() > 0:
                    avg_sentiment = self.reviews_df.loc[mask, 'sentiment'].mean()
                    phrase_data.append({
                        'phrase': phrase,
                        'count': int(count),
                        'sentiment': avg_sentiment
                    })
            
            self.common_phrases = pd.DataFrame(phrase_data).sort_values('count', ascending=False)
            
        except Exception as e:
            print(f"Common phrases extraction warning: {e}")
            self.common_phrases = pd.DataFrame()
    
    def create_visualization(self):
        """Create 4-panel topic extraction dashboard."""
        if self.reviews_df is None:
            self.load_data()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Most vs Least Positive Topics by Quarter',
                'Top Topics by Sentiment Score',
                'Most Common Phrases in Reviews',
                'Topic Sentiment Distribution'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "histogram"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # 1. Quarterly Topics - Most vs Least Positive
        if hasattr(self, 'quarterly_topics') and self.quarterly_topics is not None and len(self.quarterly_topics) > 0:
            quarters = self.quarterly_topics['quarter_label'].tolist()
            
            # Most positive (green)
            fig.add_trace(
                go.Bar(
                    x=quarters,
                    y=self.quarterly_topics['most_positive_sentiment'],
                    name='Most Positive',
                    marker_color='#00b894',
                    text=self.quarterly_topics['most_positive_topic'],
                    textposition='outside',
                    textangle=0,
                    hovertemplate='<b>%{text}</b><br>Sentiment: %{y:.2f}<br>Mentions: %{customdata}<extra></extra>',
                    customdata=self.quarterly_topics['most_positive_mentions']
                ),
                row=1, col=1
            )
            
            # Least positive (red)
            fig.add_trace(
                go.Bar(
                    x=quarters,
                    y=self.quarterly_topics['least_positive_sentiment'],
                    name='Least Positive',
                    marker_color='#ff6b6b',
                    text=self.quarterly_topics['least_positive_topic'],
                    textposition='outside',
                    textangle=0,
                    hovertemplate='<b>%{text}</b><br>Sentiment: %{y:.2f}<br>Mentions: %{customdata}<extra></extra>',
                    customdata=self.quarterly_topics['least_positive_mentions']
                ),
                row=1, col=1
            )
        
        # 2. Top Topics by Sentiment - Color gradient
        if hasattr(self, 'topic_sentiments') and len(self.topic_sentiments) > 0:
            # Get top 10 by count, then sort by sentiment
            top_topics = self.topic_sentiments.nlargest(10, 'count').sort_values('sentiment')
            
            # Color code by sentiment
            colors = []
            for sentiment in top_topics['sentiment']:
                if sentiment < -0.2:
                    colors.append('#ff6b6b')  # Red
                elif sentiment < 0:
                    colors.append('#ffa07a')  # Light red
                elif sentiment < 0.2:
                    colors.append('#98d8c8')  # Light green
                else:
                    colors.append('#00b894')  # Green
            
            fig.add_trace(
                go.Bar(
                    y=top_topics['topic'],
                    x=top_topics['sentiment'],
                    orientation='h',
                    marker_color=colors,
                    text=[f"{s:.2f}" for s in top_topics['sentiment']],
                    textposition='auto',
                    showlegend=False,
                    hovertemplate='<b>%{y}</b><br>Sentiment: %{x:.2f}<br>Mentions: %{customdata}<extra></extra>',
                    customdata=top_topics['count']
                ),
                row=1, col=2
            )
        
        # 3. Most Common Phrases
        if hasattr(self, 'common_phrases') and len(self.common_phrases) > 0:
            top_phrases = self.common_phrases.head(10)
            
            # Color by sentiment
            colors = []
            for sentiment in top_phrases['sentiment']:
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
                    y=top_phrases['phrase'],
                    x=top_phrases['count'],
                    orientation='h',
                    marker_color=colors,
                    text=top_phrases['count'],
                    textposition='outside',
                    showlegend=False,
                    hovertemplate='<b>%{y}</b><br>Mentions: %{x}<br>Avg Sentiment: %{customdata:.2f}<extra></extra>',
                    customdata=top_phrases['sentiment']
                ),
                row=2, col=1
            )
        
        # 4. Topic Sentiment Distribution
        if hasattr(self, 'topic_sentiments') and len(self.topic_sentiments) > 0:
            fig.add_trace(
                go.Histogram(
                    x=self.topic_sentiments['sentiment'],
                    nbinsx=20,
                    marker_color='#008f8c',
                    showlegend=False,
                    hovertemplate='Sentiment Range: %{x}<br>Topics: %{y}<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Add mean line
            mean_sentiment = self.topic_sentiments['sentiment'].mean()
            fig.add_vline(
                x=mean_sentiment,
                line_dash="dash",
                line_color="#023535",
                annotation_text=f"Mean: {mean_sentiment:.2f}",
                annotation_position="top",
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Topic Extraction: Themes & Sentiment Over Time",
            title_x=0.5,
            title_font=dict(size=18, color='#023535'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Quarter", row=1, col=1)
        fig.update_yaxes(title_text="Sentiment Score", row=1, col=1)
        
        fig.update_xaxes(title_text="Sentiment Score", row=1, col=2)
        fig.update_yaxes(title_text="Topic", row=1, col=2)
        
        fig.update_xaxes(title_text="Number of Mentions", row=2, col=1)
        fig.update_yaxes(title_text="Phrase", row=2, col=1)
        
        fig.update_xaxes(title_text="Sentiment Score", row=2, col=2)
        fig.update_yaxes(title_text="Number of Topics", row=2, col=2)
        
        return fig
    
    def get_insights(self) -> list:
        """Generate data-driven insights from topic analysis."""
        if self.reviews_df is None:
            self.load_data()
        
        insights = []
        
        # Quarterly trends
        if hasattr(self, 'quarterly_topics') and self.quarterly_topics is not None and len(self.quarterly_topics) > 0:
            latest = self.quarterly_topics.iloc[-1]
            insights.append(
                f"**Most recent quarter ({latest['quarter_label']})**: "
                f"Most positive topic is '{latest['most_positive_topic']}' ({latest['most_positive_sentiment']:.2f}), "
                f"least positive is '{latest['least_positive_topic']}' ({latest['least_positive_sentiment']:.2f})"
            )
            
            # Check for recurring issues
            least_positive_topics = self.quarterly_topics['least_positive_topic'].tolist()
            from collections import Counter
            topic_frequency = Counter(least_positive_topics)
            recurring = [topic for topic, count in topic_frequency.items() if count >= 2]
            
            if recurring:
                insights.append(
                    f"**Recurring issue**: '{recurring[0]}' appears as least positive in multiple quarters - "
                    f"this needs systematic attention"
                )
        
        # Common phrases insights
        if hasattr(self, 'common_phrases') and len(self.common_phrases) > 0:
            top_phrase = self.common_phrases.iloc[0]
            insights.append(
                f"Most common phrase: '{top_phrase['phrase']}' ({top_phrase['count']} mentions, "
                f"{top_phrase['sentiment']:.2f} sentiment) - "
                f"{'highlight this in marketing' if top_phrase['sentiment'] > 0.2 else 'area for improvement'}"
            )
        
        # Topic sentiment distribution
        if hasattr(self, 'topic_sentiments') and len(self.topic_sentiments) > 0:
            positive_topics = (self.topic_sentiments['sentiment'] > 0.1).sum()
            negative_topics = (self.topic_sentiments['sentiment'] < -0.1).sum()
            total_topics = len(self.topic_sentiments)
            
            insights.append(
                f"Topic sentiment balance: {positive_topics}/{total_topics} topics are positive, "
                f"{negative_topics}/{total_topics} are negative - "
                f"{'overall positive perception' if positive_topics > negative_topics else 'concerns need addressing'}"
            )
        
        # Most/least positive overall
        if hasattr(self, 'topic_sentiments') and len(self.topic_sentiments) > 0:
            best_topic = self.topic_sentiments.nlargest(1, 'sentiment').iloc[0]
            worst_topic = self.topic_sentiments.nsmallest(1, 'sentiment').iloc[0]
            
            insights.append(
                f"**Strongest asset**: '{best_topic['topic']}' ({best_topic['sentiment']:.2f} sentiment) - "
                f"mentioned {best_topic['count']} times"
            )
            
            insights.append(
                f"**Biggest opportunity**: '{worst_topic['topic']}' ({worst_topic['sentiment']:.2f} sentiment) - "
                f"addressing this could improve {worst_topic['count']} customer experiences"
            )
        
        # Connection to other analyses
        insights.append(
            "**Connection to Customer Segmentation**: Different customer segments may care about different topics - "
            "cross-reference to tailor messaging by segment"
        )
        
        return insights
    
    # Backward compatibility
    @property
    def insights(self) -> list:
        return self.get_insights()
    
    def get_recommendations(self) -> list:
        """Generate actionable recommendations."""
        recommendations = []
        
        if hasattr(self, 'quarterly_topics') and self.quarterly_topics is not None and len(self.quarterly_topics) > 0:
            # Address recurring negative topics
            least_positive_topics = self.quarterly_topics['least_positive_topic'].tolist()
            from collections import Counter
            recurring = Counter(least_positive_topics).most_common(1)[0]
            
            if recurring[1] >= 2:
                recommendations.append(
                    f"**Priority fix**: '{recurring[0]}' appears as a pain point in {recurring[1]} quarters. "
                    f"Create an action plan to systematically address this issue"
                )
        
        if hasattr(self, 'common_phrases') and len(self.common_phrases) > 0:
            # Leverage positive phrases
            positive_phrases = self.common_phrases[self.common_phrases['sentiment'] > 0.2].head(3)
            if len(positive_phrases) > 0:
                recommendations.append(
                    f"Use these phrases in marketing: {', '.join(positive_phrases['phrase'].tolist())} - "
                    f"they resonate positively with customers"
                )
            
            # Address negative phrases
            negative_phrases = self.common_phrases[self.common_phrases['sentiment'] < 0].head(2)
            if len(negative_phrases) > 0:
                recommendations.append(
                    f"Create standard responses for: {', '.join(negative_phrases['phrase'].tolist())} - "
                    f"these appear frequently in less positive reviews"
                )
        
        recommendations.append(
            "Monitor quarterly topic trends to catch emerging issues before they become widespread"
        )
        
        recommendations.append(
            "**Next step**: Use Marketing Analysis to test messaging that emphasizes Ron's strongest topics "
            f"across different channels"
        )
        
        return recommendations
    
    # Backward compatibility
    @property
    def recommendations(self) -> list:
        return self.get_recommendations()
    
    @property
    def business_impact(self) -> str:
        return "Understanding specific topics customers care about allows Ron to focus improvements on high-impact areas and craft marketing messages that resonate with actual customer experiences"
