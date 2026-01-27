"""Churn Modeling & Prediction Module - Amralytics Methodology"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .base_analysis import BaseAnalysis


class ChurnModeling(BaseAnalysis):
    """Predict customer churn and identify at-risk customers."""
    
    def __init__(self):
        super().__init__()
        self.churn_df = None
        self.model = None
        self.feature_importance = None
        self.predictions = None
        self.risk_segments = None
    
    @property
    def icon(self) -> str:
        return '⚠️'
    
    @property
    def color(self) -> str:
        return '#ff6b6b'
    
    @property
    def rons_challenge(self) -> str:
        return """Ron notices some customers stop calling after one service. Others have been loyal 
        for years but suddenly disappear. 
        
**Can we predict which customers are about to leave?** Understanding churn risk helps Ron 
proactively reach out to at-risk customers before they're gone forever."""
    
    # Backward compatibility
    @property
    def business_question(self) -> str:
        return self.rons_challenge
    
    @property
    def data_collected(self) -> list:
        return [
            'Customer service history - **ServiceTitan**',
            'Days since last service (recency) - **ServiceTitan**',
            'Service frequency patterns - **ServiceTitan**',
            'Customer lifetime value - **QuickBooks, ServiceTitan**',
            'Customer demographics (age, home age) - **ServiceTitan CRM**',
            'Churn indicators - **ServiceTitan**'
    ]
    
    # Backward compatibility
    @property
    def data_inputs(self) -> list:
        return self.data_collected
    
    @property
    def methodology(self) -> str:
        return 'Logistic regression for churn prediction, feature importance analysis, risk segmentation based on probability scores, cohort survival analysis'
    
    # Backward compatibility
    @property
    def technical_output(self) -> str:
        return self.methodology
    
    @property
    def data_file(self) -> str:
        return 'churn_modeling.csv'
    
    def engineer_features(self, df):
        """Create churn prediction features from customer data."""
        
        # Parse dates if needed
        if 'last_service_date' in df.columns:
            df['last_service_date'] = pd.to_datetime(df['last_service_date'])
            reference_date = df['last_service_date'].max()
            df['days_since_last_service'] = (reference_date - df['last_service_date']).dt.days
        
        # Calculate engagement score
        df['engagement_score'] = 0
        if 'frequency' in df.columns:
            df['engagement_score'] += df['frequency'] * 20
        if 'days_since_last_service' in df.columns:
            df['engagement_score'] -= df['days_since_last_service'] / 10
        if 'total_spend' in df.columns:
            df['engagement_score'] += df['total_spend'] / 100
        
        # Risk flags
        df['high_risk_recency'] = (df.get('days_since_last_service', 0) > 365).astype(int)
        df['low_frequency'] = (df.get('frequency', 0) <= 1).astype(int)
        df['low_spend'] = (df.get('total_spend', 0) < df.get('total_spend', pd.Series([0])).median()).astype(int)
        
        return df
    
    def train_model(self, df):
        """Train logistic regression model for churn prediction."""
        
        # Select features for modeling
        feature_cols = [
            'days_since_last_service', 'frequency', 'total_spend',
            'avg_ticket', 'customer_age', 'home_age',
            'engagement_score'
        ]
        
        # Filter to available columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        if 'churned' not in df.columns:
            # If no churn label, estimate based on recency
            df['churned'] = (df.get('days_since_last_service', 0) > 540).astype(int)
        
        X = df[available_features].fillna(0)
        y = df['churned']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_scaled, y)
        
        # Get predictions
        df['churn_probability'] = model.predict_proba(X_scaled)[:, 1]
        df['churn_prediction'] = model.predict(X_scaled)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': np.abs(model.coef_[0])
        }).sort_values('importance', ascending=False)
        
        return model, feature_importance, df
    
    def segment_by_risk(self, df):
        """Segment customers by churn risk level."""
        
        if 'churn_probability' not in df.columns:
            return None
        
        # Create risk segments
        df['risk_segment'] = pd.cut(
            df['churn_probability'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        # Summarize by segment
        risk_summary = df.groupby('risk_segment', observed=True).agg({
            'customer_id': 'count',
            'total_spend': 'mean',
            'days_since_last_service': 'mean',
            'frequency': 'mean'
        }).reset_index()
        
        risk_summary.columns = [
            'risk_segment', 'customer_count', 'avg_spend',
            'avg_days_since', 'avg_frequency'
        ]
        
        return risk_summary
    
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """Load and process churn modeling data."""
        if filepath is None:
            filepath = f'data/{self.data_file}'
        
        print(f"Loading churn data from: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df)} customer records")
        except FileNotFoundError:
            print(f"⚠️ {filepath} not found - creating synthetic churn data")
            # Create minimal synthetic data for demo
            df = self._create_synthetic_churn_data()
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Train model
        self.model, self.feature_importance, df = self.train_model(df)
        
        # Segment by risk
        self.risk_segments = self.segment_by_risk(df)
        
        self.churn_df = df
        self.data = df
        return df
    
    def _create_synthetic_churn_data(self):
        """Create synthetic churn data if file doesn't exist."""
        np.random.seed(42)
        n_customers = 100
        
        df = pd.DataFrame({
            'customer_id': [f'CUST_{i:03d}' for i in range(1, n_customers + 1)],
            'days_since_last_service': np.random.randint(30, 900, n_customers),
            'frequency': np.random.randint(1, 8, n_customers),
            'total_spend': np.random.uniform(200, 5000, n_customers),
            'avg_ticket': np.random.uniform(150, 800, n_customers),
            'customer_age': np.random.randint(30, 75, n_customers),
            'home_age': np.random.randint(5, 50, n_customers),
        })
        
        # Create churn labels based on recency and frequency
        df['churned'] = (
            ((df['days_since_last_service'] > 540) & (df['frequency'] <= 2)) |
            (df['days_since_last_service'] > 720)
        ).astype(int)
        
        return df
    
    def create_visualization(self):
        """Create 4-panel churn modeling dashboard."""
        if self.churn_df is None:
            self.load_data()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Churn Risk Distribution',
                'Feature Importance for Churn',
                'Risk Segments: Customer Count & Value',
                'Days Since Last Service vs Churn Probability'
            ),
            specs=[
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.22,
            horizontal_spacing=0.15
        )
        
        # Color mapping
        risk_colors = {
            'Low Risk': '#00b894',
            'Medium Risk': '#ffa07a',
            'High Risk': '#ff6b6b'
        }
        
        # 1. Churn Probability Distribution
        fig.add_trace(
            go.Histogram(
                x=self.churn_df['churn_probability'],
                nbinsx=30,
                marker_color='#008f8c',
                name='Customers',
                showlegend=False,
                hovertemplate='Churn Probability: %{x:.2f}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add risk threshold lines
        fig.add_vline(x=0.3, line_dash="dash", line_color="#00b894", opacity=0.5, row=1, col=1)
        fig.add_vline(x=0.6, line_dash="dash", line_color="#ff6b6b", opacity=0.5, row=1, col=1)
        
        # 2. Feature Importance
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(8)
            
            fig.add_trace(
                go.Bar(
                    y=top_features['feature'],
                    x=top_features['importance'],
                    orientation='h',
                    marker_color='#23606e',
                    text=[f"{v:.2f}" for v in top_features['importance']],
                    textposition='outside',
                    showlegend=False,
                    hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. Risk Segments
        if self.risk_segments is not None:
            # Customer count bars
            colors_segment = [risk_colors.get(str(seg), '#023535') for seg in self.risk_segments['risk_segment']]
            
            fig.add_trace(
                go.Bar(
                    x=self.risk_segments['risk_segment'].astype(str),
                    y=self.risk_segments['customer_count'],
                    marker_color=colors_segment,
                    text=self.risk_segments['customer_count'],
                    textposition='outside',
                    name='Customers',
                    showlegend=False,
                    hovertemplate='<b>%{x}</b><br>Customers: %{y}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Recency vs Churn Probability Scatter
        # Color by risk segment
        for risk_level in ['Low Risk', 'Medium Risk', 'High Risk']:
            segment_data = self.churn_df[self.churn_df['risk_segment'] == risk_level]
            
            fig.add_trace(
                go.Scatter(
                    x=segment_data['days_since_last_service'],
                    y=segment_data['churn_probability'],
                    mode='markers',
                    name=risk_level,
                    marker=dict(
                        size=6,
                        color=risk_colors.get(risk_level, '#023535'),
                        opacity=0.6,
                        line=dict(width=0.5, color='white')
                    ),
                    hovertemplate=(
                        f'<b>{risk_level}</b><br>'
                        'Days Since: %{x}<br>'
                        'Churn Prob: %{y:.2f}<extra></extra>'
                    )
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text="Churn Prediction: Identifying At-Risk Customers",
            title_x=0.5,
            title_font=dict(size=18, color='#023535'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=100, r=80, t=100, b=80)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Churn Probability", row=1, col=1)
        fig.update_yaxes(title_text="Number of Customers", row=1, col=1)
        
        fig.update_xaxes(title_text="Feature Importance", row=1, col=2)
        fig.update_yaxes(title_text="Feature", row=1, col=2)
        
        fig.update_xaxes(title_text="Risk Segment", row=2, col=1, tickangle=-45)
        fig.update_yaxes(title_text="Customer Count", row=2, col=1)
        
        fig.update_xaxes(title_text="Days Since Last Service", row=2, col=2)
        fig.update_yaxes(title_text="Churn Probability", row=2, col=2)
        
        return fig
    
    def get_insights(self) -> list:
        """Generate data-driven insights from churn analysis."""
        if self.churn_df is None:
            self.load_data()
        
        insights = []
        
        # Overall churn rate
        churn_rate = self.churn_df['churned'].mean() * 100
        insights.append(
            f"**Overall churn rate**: {churn_rate:.1f}% of customers have churned "
            f"({self.churn_df['churned'].sum()} out of {len(self.churn_df)})"
        )
        
        # High risk customers
        high_risk = self.churn_df[self.churn_df['churn_probability'] > 0.6]
        if len(high_risk) > 0:
            high_risk_revenue = high_risk['total_spend'].sum()
            insights.append(
                f"**{len(high_risk)} customers at high risk** (>60% churn probability) - "
                f"${high_risk_revenue:,.0f} in lifetime value at stake"
            )
        
        # Top churn driver
        if self.feature_importance is not None and len(self.feature_importance) > 0:
            top_feature = self.feature_importance.iloc[0]
            insights.append(
                f"**Primary churn driver**: '{top_feature['feature']}' "
                f"(importance: {top_feature['importance']:.3f}) - focus interventions here"
            )
        
        # Risk segment breakdown
        if self.risk_segments is not None:
            high_risk_seg = self.risk_segments[self.risk_segments['risk_segment'] == 'High Risk']
            if len(high_risk_seg) > 0:
                high_risk_count = high_risk_seg['customer_count'].values[0]
                high_risk_pct = (high_risk_count / len(self.churn_df)) * 100
                insights.append(
                    f"**Risk distribution**: {high_risk_pct:.0f}% of customers in high-risk segment - "
                    f"immediate action needed"
                )
        
        # Recency insight
        avg_days_churned = self.churn_df[self.churn_df['churned'] == 1]['days_since_last_service'].mean()
        avg_days_active = self.churn_df[self.churn_df['churned'] == 0]['days_since_last_service'].mean()
        
        insights.append(
            f"**Recency matters**: Churned customers averaged {avg_days_churned:.0f} days since last service "
            f"vs {avg_days_active:.0f} days for active customers"
        )
        
        # Model performance
        if self.model is not None:
            churn_prob_avg = self.churn_df['churn_probability'].mean()
            insights.append(
                f"Model identifies customers with average {churn_prob_avg*100:.0f}% churn probability - "
                f"use for prioritizing outreach"
            )
        
        # Connection to other analyses
        insights.append(
            "**Connection to Customer Segmentation**: Different segments have different churn patterns - "
            "tailor retention strategies by segment"
        )
        
        return insights
    
    # Backward compatibility
    @property
    def insights(self) -> list:
        return self.get_insights()
    
    def get_recommendations(self) -> list:
        """Generate actionable recommendations."""
        recommendations = []
        
        if self.churn_df is not None:
            # High risk outreach
            high_risk = self.churn_df[self.churn_df['churn_probability'] > 0.6]
            if len(high_risk) > 0:
                recommendations.append(
                    f"**Immediate outreach to {len(high_risk)} high-risk customers**: "
                    f"Call within 48 hours with special offer or check-in"
                )
            
            # Feature-specific actions
            if self.feature_importance is not None:
                top_feature = self.feature_importance.iloc[0]['feature']
                
                if 'days_since' in top_feature.lower():
                    recommendations.append(
                        "**Proactive scheduling**: Set up automated reminders at 90, 180, 270 days - "
                        "reach customers before they churn"
                    )
                elif 'frequency' in top_feature.lower():
                    recommendations.append(
                        "**Convert to maintenance plans**: One-time customers are highest risk - "
                        "offer annual maintenance contracts"
                    )
            
            # Segment-specific
            recommendations.append(
                "Create retention campaigns by risk level: High risk gets personal call, "
                "medium risk gets special offer email, low risk gets newsletter"
            )
            
            # Win-back for churned
            churned_count = self.churn_df['churned'].sum()
            if churned_count > 0:
                recommendations.append(
                    f"**Win-back campaign for {churned_count} churned customers**: "
                    f"'We Miss You' discount on seasonal tune-up"
                )
        
        recommendations.append(
            "Track churn by customer segment (from Segmentation Analysis) - "
            "different segments need different retention approaches"
        )
        
        recommendations.append(
            "**Next step**: Use Sentiment Analysis to understand why customers leave - "
            "address root causes, not just symptoms"
        )
        
        return recommendations
    
    # Backward compatibility
    @property
    def recommendations(self) -> list:
        return self.get_recommendations()
    
    @property
    def business_impact(self) -> str:
        return "Reducing churn by 5% can increase customer lifetime value by 25-95%. Early intervention with at-risk customers prevents revenue loss and protects Ron's customer base."
