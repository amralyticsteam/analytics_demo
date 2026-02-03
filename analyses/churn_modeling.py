"""Churn Modeling & Prediction Module - Amralytics Methodology"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
        return """Ron's client base is aging. Customers who have been loyal for years are suddenly moving house. He knows he can convince some clients to give referrals, or move their service to a new house. He has limited time with which to assess which clients to reach out to. He needs to identify which clients are at risk of leaving so he can prioritize his calls.

Can we predict which customers are about to leave? Understanding **churn risk** helps Ron proactively reach out to at-risk customers before they're gone forever."""
    
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
    
    @property
    def data_inputs(self) -> list:
        return self.data_collected
    
    @property
    def methodology(self) -> str:
        return """We use the following analytical techniques to provide Ron with key indicators of at-risk clients, as well as a list of people to call:

**Logistic regression** - Think of it like a credit score, but for customer churn risk. It looks at patterns in customer behavior (visit frequency, time since last service, satisfaction) and assigns each customer a risk score from 0-100%.

**Feature importance analysis** shows us which factors matter most - is it recency? Frequency? Complaints? This tells Ron where to focus his retention efforts.

**Risk segmentation** groups customers into Low/Medium/High risk categories, making it easy to create targeted campaigns (high-risk gets a phone call, medium-risk gets an email offer).

**Why this works for Ron:** It tells us not just WHO might leave, but WHY (which factors drive churn), so Ron can take targeted action.

**If results aren't strong enough, we could:**
- Try more sophisticated models (Random Forest, XGBoost) that catch complex patterns
- Add more data sources (sentiment from reviews, seasonal patterns, competitor activity)
- Use survival analysis to predict when customers will churn, not just if they will
- Build customer lifetime value models to prioritize which at-risk customers are worth saving"""

    @property
    def technical_output(self) -> str:
        return self.methodology
    
    @property
    def data_file(self) -> str:
        return 'churn_modeling.csv'
    
    def _create_synthetic_churn_data(self):
        """Create synthetic churn data."""
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
        
        # Create realistic churn labels
        df['churned'] = (
            ((df['days_since_last_service'] > 540) & (df['frequency'] <= 2)) |
            (df['days_since_last_service'] > 720)
        ).astype(int)
        
        return df
    
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """Load and process churn modeling data."""
        if filepath is None:
            filepath = f'data/{self.data_file}'
        
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Loaded {len(df)} customers from {filepath}")
            print(f"✓ Columns: {list(df.columns)}")
        except FileNotFoundError:
            print(f"⚠️  File not found: {filepath}")
            print(f"⚠️  Creating synthetic data instead")
            df = self._create_synthetic_churn_data()
            print(f"✓ Created {len(df)} synthetic customers")
        
        # Map column names to standard names if needed
        column_mapping = {
            'months_since_last_service': 'months_since_last',
            'total_lifetime_value': 'total_spend',
            'service_count': 'frequency',
            'avg_ticket_size': 'avg_ticket',
            'churn_label': 'churned'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Convert months to days if needed
        if 'months_since_last' in df.columns:
            df['days_since_last_service'] = df['months_since_last'] * 30
        
        # Ensure churned column exists
        if 'churned' not in df.columns:
            # Create churn labels based on recency and frequency
            if 'days_since_last_service' in df.columns and 'frequency' in df.columns:
                df['churned'] = (
                    ((df['days_since_last_service'] > 540) & (df['frequency'] <= 2)) |
                    (df['days_since_last_service'] > 720)
                ).astype(int)
                print(f"✓ Created churn labels: {df['churned'].sum()} churned, {(~df['churned'].astype(bool)).sum()} active")
            else:
                np.random.seed(42)
                df['churned'] = (np.random.random(len(df)) < 0.25).astype(int)
                print(f"✓ Created synthetic churn labels")
        else:
            print(f"✓ Using existing churn labels: {df['churned'].sum()} churned, {(~df['churned'].astype(bool)).sum()} active")
        
        # Ensure we have both classes
        if df['churned'].sum() == 0:
            print(f"⚠️  No churned customers - adding some synthetic churn")
            np.random.seed(42)
            df.loc[df.sample(frac=0.2, random_state=42).index, 'churned'] = 1
            print(f"✓ Now have {df['churned'].sum()} churned customers")
        
        # Train model with available features
        feature_cols = [
            'days_since_last_service', 'frequency', 'total_spend', 'avg_ticket',
            'customer_age', 'home_age', 'complaint_count', 'response_satisfaction',
            'price_sensitivity'
        ]
        
        # Handle categorical variables BEFORE selecting features
        if 'price_sensitivity' in df.columns and df['price_sensitivity'].dtype == 'object':
            # Convert Low/Medium/High to numeric
            price_map = {'Low': 1, 'Medium': 2, 'High': 3}
            df['price_sensitivity'] = df['price_sensitivity'].map(price_map).fillna(2)
            print(f"✓ Converted price_sensitivity to numeric")
        
        available_features = [col for col in feature_cols if col in df.columns]
        print(f"✓ Available features: {available_features}")
        
        if len(available_features) >= 2:
            # Select features and handle each column appropriately
            X = df[available_features].copy()
            
            # Fill missing values column by column
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    # For any remaining non-numeric columns, convert or fill with mode
                    print(f"⚠️  Column {col} is {X[col].dtype} - converting to numeric")
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
            y = df['churned']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            self.model.fit(X_scaled, y)
            print(f"✓ Model trained successfully")
            
            df['churn_probability'] = self.model.predict_proba(X_scaled)[:, 1]
            df['churn_prediction'] = self.model.predict(X_scaled)
            print(f"✓ Predictions created: avg probability = {df['churn_probability'].mean():.2f}")
            print(f"  Probability range: {df['churn_probability'].min():.2f} - {df['churn_probability'].max():.2f}")
            
            self.feature_importance = pd.DataFrame({
                'feature': available_features,
                'importance': np.abs(self.model.coef_[0])
            }).sort_values('importance', ascending=False)
            print(f"✓ Feature importance calculated")
            print(f"  Top feature: {self.feature_importance.iloc[0]['feature']}")
            
            # Create risk segments
            df['risk_segment'] = pd.cut(
                df['churn_probability'],
                bins=[0, 0.3, 0.6, 1.0],
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
            print(f"✓ Risk segments created")
            
            # Risk summary
            self.risk_segments = df.groupby('risk_segment', observed=True).size().reset_index(name='customer_count')
            print(f"✓ Risk summary:")
            for _, row in self.risk_segments.iterrows():
                print(f"    {row['risk_segment']}: {row['customer_count']} customers")
        else:
            print(f"❌ Not enough features for modeling (need 2+, have {len(available_features)})")
        
        self.churn_df = df
        self.data = df
        print(f"✓ Data loading complete - {len(df)} customers ready")
        return df
    
    def create_visualization(self):
        """Create 4-panel churn dashboard."""
        if self.churn_df is None:
            print("⚠️  churn_df is None, loading data...")
            self.load_data()
        
        # Debug: Check what we have
        print(f"DEBUG: churn_df shape: {self.churn_df.shape if self.churn_df is not None else 'None'}")
        print(f"DEBUG: churn_df columns: {list(self.churn_df.columns) if self.churn_df is not None else 'None'}")
        print(f"DEBUG: feature_importance: {self.feature_importance is not None}")
        print(f"DEBUG: risk_segments: {self.risk_segments is not None}")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Churn Risk Distribution',
                'Feature Importance for Churn',
                'Risk Segments: Customer Count',
                'Days Since Last Service vs Churn Probability'
            ),
            specs=[
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.22,
            horizontal_spacing=0.15
        )
        
        risk_colors = {'Low Risk': '#00b894', 'Medium Risk': '#ffa07a', 'High Risk': '#ff6b6b'}
        
        # 1. Churn Probability Distribution
        if self.churn_df is not None and 'churn_probability' in self.churn_df.columns:
            print("✓ Adding histogram")
            fig.add_trace(go.Histogram(
                x=self.churn_df['churn_probability'],
                nbinsx=30,
                marker_color='#008f8c',
                name='Customers',
                showlegend=False,
                hovertemplate='Churn Probability: %{x:.2f}<br>Count: %{y}<extra></extra>'
            ), row=1, col=1)
            fig.add_vline(x=0.3, line_dash="dash", line_color="#00b894", opacity=0.5, row=1, col=1)
            fig.add_vline(x=0.6, line_dash="dash", line_color="#ff6b6b", opacity=0.5, row=1, col=1)
        else:
            print("❌ Cannot add histogram - missing churn_probability")
        
        # 2. Feature Importance
        if self.feature_importance is not None and len(self.feature_importance) > 0:
            print(f"✓ Adding feature importance ({len(self.feature_importance)} features)")
            fig.add_trace(go.Bar(
                y=self.feature_importance['feature'],
                x=self.feature_importance['importance'],
                orientation='h',
                marker_color='#23606e',
                text=[f"{v:.2f}" for v in self.feature_importance['importance']],
                textposition='outside',
                showlegend=False,
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
            ), row=1, col=2)
        else:
            print("❌ Cannot add feature importance - missing data")
        
        # 3. Risk Segments
        if self.risk_segments is not None and len(self.risk_segments) > 0:
            print(f"✓ Adding risk segments ({len(self.risk_segments)} segments)")
            colors = [risk_colors.get(str(s), '#023535') for s in self.risk_segments['risk_segment']]
            fig.add_trace(go.Bar(
                x=self.risk_segments['risk_segment'].astype(str),
                y=self.risk_segments['customer_count'],
                marker_color=colors,
                text=self.risk_segments['customer_count'],
                textposition='outside',
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Customers: %{y}<extra></extra>'
            ), row=2, col=1)
        else:
            print("❌ Cannot add risk segments - missing data")
        
        # 4. Scatter plot
        if (self.churn_df is not None and 
            'risk_segment' in self.churn_df.columns and 
            'days_since_last_service' in self.churn_df.columns and
            'churn_probability' in self.churn_df.columns):
            print("✓ Adding scatter plot")
            for risk in ['Low Risk', 'Medium Risk', 'High Risk']:
                segment = self.churn_df[self.churn_df['risk_segment'] == risk]
                if len(segment) > 0:
                    fig.add_trace(go.Scatter(
                        x=segment['days_since_last_service'],
                        y=segment['churn_probability'],
                        mode='markers',
                        name=risk,
                        marker=dict(size=6, color=risk_colors[risk], opacity=0.6)
                    ), row=2, col=2)
        
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text="Churn Prediction: Identifying At-Risk Customers",
            title_x=0.5,
            margin=dict(l=100, r=80, t=100, b=80)
        )
        
        fig.update_xaxes(title_text="Churn Probability", row=1, col=1)
        fig.update_yaxes(title_text="Number of Customers", row=1, col=1, rangemode='tozero')
        fig.update_xaxes(title_text="Feature Importance", row=1, col=2)
        fig.update_xaxes(title_text="Risk Segment", row=2, col=1)
        fig.update_yaxes(title_text="Customer Count", row=2, col=1, rangemode='tozero')
        fig.update_xaxes(title_text="Days Since Last Service", row=2, col=2)
        fig.update_yaxes(title_text="Churn Probability", row=2, col=2, rangemode='tozero')
        
        return fig
    
    def get_insights(self) -> list:
        """Generate insights."""
        if self.churn_df is None:
            self.load_data()
        
        insights = []
        
        if 'churned' in self.churn_df.columns:
            churn_rate = self.churn_df['churned'].mean() * 100
            insights.append(f"Overall churn rate: {churn_rate:.1f}% of customers have churned")
        
        if 'churn_probability' in self.churn_df.columns:
            high_risk = self.churn_df[self.churn_df['churn_probability'] > 0.6]
            insights.append(f"{len(high_risk)} customers at high risk (>60% churn probability)")
        
        if self.feature_importance is not None and len(self.feature_importance) > 0:
            top = self.feature_importance.iloc[0]
            insights.append(f"Primary churn driver: '{top['feature']}' - focus interventions here")
        
        insights.append("Connection to Customer Segmentation: Different segments have different churn patterns")
        
        return insights
    
    @property
    def insights(self) -> list:
        return self.get_insights()
    
    def get_recommendations(self) -> list:
        """Generate actionable churn prevention recommendations."""
        if self.churn_df is None:
            self.load_data()
        
        recs = []
        
        # Calculate key metrics
        total_customers = len(self.churn_df)
        
        # Use churn_label column
        if 'churn_label' in self.churn_df.columns:
            high_risk = self.churn_df[self.churn_df['churn_label'] == 1]
            high_risk_count = len(high_risk)
            active_count = total_customers - high_risk_count
            
            if high_risk_count > 0:
                recs.append(
                    f"Launch win-back campaign immediately: Contact {high_risk_count} at-risk customers "
                    f"({high_risk_count/total_customers*100:.1f}% of base) with seasonal tune-up offers. "
                    f"Target 25% win-back rate = ~${int(high_risk_count * 0.25 * 2000):,} preserved revenue"
                )
                
                # Average recency for high-risk customers
                if 'months_since_last_service' in self.churn_df.columns and len(high_risk) > 0:
                    avg_recency = high_risk['months_since_last_service'].mean()
                    recs.append(
                        f"Set up automated outreach triggers: Customers inactive {avg_recency:.0f}+ months are high risk. "
                        f"Implement automated email/SMS at 6, 9, and 12 months with increasing urgency"
                    )
            else:
                recs.append(
                    f"Strong retention! Only {high_risk_count} customers at risk. "
                    f"Maintain current service quality and continue proactive outreach"
                )
        
        # Service frequency recommendations
        if 'service_count' in self.churn_df.columns:
            low_frequency = self.churn_df[self.churn_df['service_count'] < 2]
            if len(low_frequency) > 0:
                recs.append(
                    f"**Build frequency with maintenance contracts**: {len(low_frequency)} customers "
                    f"({len(low_frequency)/total_customers*100:.0f}%) have only 1 service. "
                    f"Offer annual maintenance plans to increase engagement and predictability"
                )
        
        # Complaint-based recommendations
        if 'complaint_count' in self.churn_df.columns:
            has_complaints = self.churn_df[self.churn_df['complaint_count'] > 0]
            if len(has_complaints) > 0:
                recs.append(
                    f"Follow up on service issues: {len(has_complaints)} customers "
                    f"({len(has_complaints)/total_customers*100:.0f}%) had complaints. "
                    f"Personal call from Ron to resolve issues can save relationships worth $1,500+ each"
                )
        
        # Referral source insights
        if 'referral_source' in self.churn_df.columns and 'churn_label' in self.churn_df.columns:
            source_churn = self.churn_df.groupby('referral_source')['churn_label'].mean()
            if len(source_churn) > 0 and source_churn.max() > 0:
                worst_source = source_churn.idxmax()
                worst_rate = source_churn.max() * 100
                best_source = source_churn.idxmin()
                best_rate = source_churn.min() * 100
                
                if worst_rate > best_rate:
                    recs.append(
                        f"Improve onboarding for {worst_source} customers: They churn at {worst_rate:.0f}% vs "
                        f"{best_source} at {best_rate:.0f}%. Add extra touchpoints (welcome call, 30-day check-in) "
                        f"for {worst_source} customers to build loyalty early"
                    )
        
        # General best practices (always include these)
        recs.append(
            "Create quarterly 'check-in' campaigns: Even if system is fine, a courtesy call/email "
            "keeps Ron top-of-mind when something does break"
        )
        
        recs.append(
            "Track prevention metrics: Monitor monthly how many at-risk customers were successfully retained. "
            "Goal: Keep churn rate under 10% annually"
        )
        
        return recs
    
    @property
    def recommendations(self) -> list:
        return self.get_recommendations()
    
    @property
    def business_impact(self) -> str:
        if self.churn_df is not None:
            total_customers = len(self.churn_df)
            churned = (self.churn_df['churn_label'] == 1).sum()
            churn_rate = (churned / total_customers) * 100
            
            # Calculate potential revenue saved
            avg_ltv = self.churn_df['total_lifetime_value'].mean() if 'total_lifetime_value' in self.churn_df.columns else 2000
            
            # If we reduce churn by 5 percentage points
            customers_saved = int(total_customers * 0.05)
            revenue_saved = customers_saved * avg_ltv
            
            return (f"Ron has {churned} at-risk customers ({churn_rate:.1f}% churn rate). "
                   f"Reducing churn by just 5 percentage points would save ~{customers_saved} customers, "
                   f"preserving ${revenue_saved:,.0f} in lifetime value.")
        
        return "Ron gets a clear list of customers who haven't called in over a year and are likely gone for good. He can reach out with a 'we miss you' offer before it's too late. Winning back old customers is way cheaper than finding new ones."
