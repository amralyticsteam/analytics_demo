"""Customer Segmentation & Clustering Module - HVAC Business Best Practices"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .base_analysis import BaseAnalysis


class CustomerSegmentation(BaseAnalysis):
    """Cluster HVAC customers using RFM + behavioral segmentation."""
    
    def __init__(self):
        super().__init__()
        self.transactions_df = None
        self.customer_features = None
        self.X_scaled = None
        self.pca_result = None
        self.pca = None
        self.labels = None
        self.n_clusters = 0
        self.segment_profiles = None
    
    @property
    def icon(self) -> str:
        return '👥'
    
    @property
    def color(self) -> str:
        return '#23606e'
    
    @property
    def rons_challenge(self) -> str:
        return """Ron treats all customers the same way, but are they really all the same? 
        
Some call only for emergencies. Others schedule regular maintenance. Some spend thousands 
on installations, others just need tune-ups. **Understanding customer segments** helps Ron 
tailor his marketing, pricing, and service approach to each group's specific needs."""
    
    # Backward compatibility
    @property
    def business_question(self) -> str:
        return self.rons_challenge
    
    @property
    def data_collected(self) -> list:
        return [
            'Customer transaction history (1,778 transactions) - **ServiceTitan**',
            'RFM metrics (Recency, Frequency, Monetary) - **ServiceTitan**',
            'Service mix preferences by customer - **ServiceTitan**',
            'Customer tenure and engagement patterns - **ServiceTitan**',
            'Payment and timing behaviors - **ServiceTitan**'
    ]
    
    # Backward compatibility
    @property
    def data_inputs(self) -> list:
        return self.data_collected
    
    @property
    def methodology(self) -> str:
        return 'RFM (Recency, Frequency, Monetary) analysis combined with service preference clustering using K-Means, customer lifetime value calculation, segment profiling'
    
    # Backward compatibility
    @property
    def technical_output(self) -> str:
        return self.methodology
    
    @property
    def data_file(self) -> str:
        return 'customer_segmentation_transactions.csv'
    
    def calculate_rfm_features(self, df):
        """Calculate RFM and behavioral features for each customer."""
        
        # Parse dates
        df["date"] = pd.to_datetime(df["date"])
        reference_date = df["date"].max() + pd.Timedelta(days=1)
        
        # Calculate RFM metrics
        rfm = df.groupby("customer_id").agg({
            "date": lambda x: (reference_date - x.max()).days,  # Recency
            "amount": ["sum", "mean", "count"]  # Monetary, avg ticket, Frequency
        }).reset_index()
        
        rfm.columns = ['customer_id', 'recency_days', 'total_spend', 'avg_ticket', 'frequency']
        
        service_mix = df.groupby("customer_id")["service_category"].value_counts().unstack(fill_value=0)
        service_mix = service_mix.div(service_mix.sum(axis=1), axis=0) * 100  # Convert to percentages
        
        # Ensure all categories exist
        for cat in ['Installation', 'Cooling', 'Heating', 'Maintenance', 'Emergency']:
            if cat not in service_mix.columns:
                service_mix[cat] = 0
        
        service_mix.columns = [f'pct_{col.lower()}' for col in service_mix.columns]
        service_mix = service_mix.reset_index()
        
        # Calculate additional behavioral metrics
        customer_behavior = df.groupby("customer_id").agg({
            "duration_hours": "mean",
            "parts_cost": "mean",
            "date": ["min", "max"]
        }).reset_index()
        
        customer_behavior.columns = ['customer_id', 'avg_duration', 'avg_parts_cost', 'first_purchase', 'last_purchase']
        customer_behavior['customer_tenure_days'] = (customer_behavior['last_purchase'] - customer_behavior['first_purchase']).dt.days
        customer_behavior['is_recent'] = (customer_behavior['customer_tenure_days'] < 90).astype(int)
        
        # Merge everything
        customer_features = rfm.merge(service_mix, on='customer_id', how='left')
        customer_features = customer_features.merge(
            customer_behavior[['customer_id', 'avg_duration', 'avg_parts_cost', 'customer_tenure_days', 'is_recent']], 
            on='customer_id', 
            how='left'
        )
        
        # Fill missing service percentages with 0
        for col in customer_features.columns:
            if col.startswith('pct_'):
                customer_features[col] = customer_features[col].fillna(0)
        
        return customer_features
    
    def perform_clustering(self, customer_features):
        """Apply K-Means clustering for clear segment identification."""
        
        # Select features for clustering
        feature_cols = [
            'recency_days', 'frequency', 'total_spend', 'avg_ticket',
            'pct_installation', 'pct_cooling', 'pct_heating', 'pct_maintenance',
            'avg_duration', 'customer_tenure_days'
        ]
        
        # Ensure all columns exist
        available_cols = [col for col in feature_cols if col in customer_features.columns]
        X = customer_features[available_cols].copy()
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters (use 4-6 for interpretability)
        n_clusters = 5
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # PCA for visualization
        pca = PCA(n_components=min(3, len(available_cols)), random_state=42)
        pca_result = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(
            pca_result,
            columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]
        )
        
        # Store feature importance (PCA loadings for first 3 components)
        self.feature_importance = pd.DataFrame(
            pca.components_[:3].T,
            columns=['PC1', 'PC2', 'PC3'],
            index=available_cols
        )
        
        return X_scaled, pca_df, pca, labels, n_clusters, available_cols
    
    def create_segment_profiles(self, customer_features, labels, transactions_df):
        """Create detailed profiles for each segment with descriptive names."""
        profiles = []
        
        for segment_id in sorted(set(labels)):
            segment_data = customer_features[labels == segment_id]
            
            if len(segment_data) == 0:
                continue
            
            # Calculate profile metrics
            profile = {
                'segment_id': segment_id,
                'size': len(segment_data),
                'avg_recency': segment_data['recency_days'].mean(),
                'avg_frequency': segment_data['frequency'].mean(),
                'avg_total_spend': segment_data['total_spend'].mean(),
                'avg_ticket': segment_data['avg_ticket'].mean(),
                'total_revenue': segment_data['total_spend'].sum(),
            }
            
            # Calculate service mix percentages from actual transactions
            segment_customer_ids = segment_data['customer_id'].tolist()
            segment_transactions = transactions_df[transactions_df['customer_id'].isin(segment_customer_ids)]
            
            # Count service categories for this segment
            if len(segment_transactions) > 0:
                service_counts = segment_transactions['service_category'].value_counts()
                total_services = service_counts.sum()
                
                # Calculate percentages that sum to 100
                for cat in ['Installation', 'Cooling', 'Heating', 'Maintenance', 'Emergency']:
                    pct_col = f'pct_{cat.lower()}'
                    if cat in service_counts.index:
                        profile[pct_col] = (service_counts[cat] / total_services) * 100
                    else:
                        profile[pct_col] = 0
            else:
                # No transactions, set all to 0
                for cat in ['Installation', 'Cooling', 'Heating', 'Maintenance', 'Emergency']:
                    profile[f'pct_{cat.lower()}'] = 0
            
            profile['avg_tenure'] = segment_data.get('customer_tenure_days', pd.Series([0])).mean()
            
            # Assign descriptive names based on characteristics
            # Add segment_id to ensure uniqueness
            base_name = ''
            
            # Segment naming based on combined RFM metrics
            recency = profile['avg_recency']
            frequency = profile['avg_frequency']
            total_spend = profile['avg_total_spend']
            
            # Define thresholds
            high_spend_threshold = customer_features['total_spend'].quantile(0.75)  # ~$4000+
            high_freq_threshold = 7  # 7+ services
            active_recency = 60  # Last 60 days
            dormant_recency = 180  # 180+ days inactive
            
            # Assign names based on clear patterns
            if frequency >= 15 and total_spend >= high_spend_threshold:
                # Segment 1: Very high frequency + high spend
                profile['name'] = 'VIP Loyal Customers'
            elif frequency >= high_freq_threshold and total_spend >= high_spend_threshold:
                # Segment 2: High frequency + high spend
                profile['name'] = 'Premium Service Contracts'
            elif recency <= active_recency and frequency >= 3:
                # Segment 0: Active, decent frequency
                profile['name'] = 'Active Regulars'
            elif recency >= dormant_recency:
                # Segment 3 or 4: Inactive/dormant
                if total_spend < customer_features['total_spend'].median():
                    profile['name'] = 'At-Risk One-Timers'
                else:
                    profile['name'] = 'Dormant Former Regulars'
            else:
                # Remaining customers - moderate engagement
                profile['name'] = 'Moderate Service Users'
            
            # Make name unique by checking if it already exists
            existing_names = [p['name'] for p in profiles]
            if base_name in existing_names:
                # Add a distinguisher based on key characteristic
                if 'Occasional' in base_name:
                    # Differentiate by spend level
                    if profile['avg_total_spend'] > customer_features['total_spend'].median():
                        profile['name'] = 'Occasional Service (Higher Spend)'
                    else:
                        profile['name'] = 'Occasional Service (Lower Spend)'
                elif 'Recent' in base_name:
                    if profile['avg_frequency'] > customer_features['frequency'].median():
                        profile['name'] = 'Recent Customers (Frequent)'
                    else:
                        profile['name'] = 'Recent Customers (New)'
                else:
                    # Fallback: add segment number
                    profile['name'] = f"{base_name} #{segment_id}"
            else:
                profile['name'] = base_name
            
            profiles.append(profile)
        
        return pd.DataFrame(profiles)
    
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """Load and process HVAC transaction data."""
        if filepath is None:
            filepath = f'data/{self.data_file}'
        
        print(f"Loading transaction data from: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} transactions")
        
        # Calculate RFM features
        self.customer_features = self.calculate_rfm_features(df)
        print(f"Calculated features for {len(self.customer_features)} unique customers")
        
        # Cluster
        self.X_scaled, self.pca_result, self.pca, self.labels, self.n_clusters, self.feature_cols = \
            self.perform_clustering(self.customer_features)
        
        # Create segment profiles
        self.segment_profiles = self.create_segment_profiles(self.customer_features, self.labels, df)
        print(f"\nFound {self.n_clusters} customer segments:")
        for _, seg in self.segment_profiles.iterrows():
            print(f"  {seg['name']}: {seg['size']} customers, ${seg['avg_total_spend']:,.0f} avg LTV")
        
        self.transactions_df = df
        self.data = self.customer_features
        return self.customer_features
    
    def create_visualization(self):
        """Create comprehensive customer segmentation dashboard."""
        if self.customer_features is None:
            self.load_data()
        
        # Check if we have PCA results for 3D viz
        has_3d = self.pca_result is not None and self.pca_result.shape[1] >= 3
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '3D Customer Clustering (PCA Components)',
                'Segment Size & Average Customer Value',
                'Service Mix by Segment',
                'RFM Profile by Segment'
            ),
            specs=[
                [{"type": "scatter3d"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.20,
            horizontal_spacing=0.15
        )
        
        # Color mapping
        color_map = {
            0: '#008f8c',
            1: '#00b894',
            2: '#23606e',
            3: '#ffa07a',
            4: '#98d8c8',
            5: '#015958'
        }
        
        # Add segment column to customer features
        self.customer_features['segment'] = self.labels
        
        # 1. 3D Clustering Visualization (PCA)
        if has_3d:
            for segment_id in sorted(set(self.labels)):
                segment_data = self.customer_features[self.customer_features['segment'] == segment_id]
                segment_name = self.segment_profiles[self.segment_profiles['segment_id'] == segment_id]['name'].values[0]
                
                # Get PCA coordinates for this segment
                segment_indices = segment_data.index
                pca_segment = self.pca_result.loc[segment_indices]
                
                fig.add_trace(
                    go.Scatter3d(
                        x=pca_segment['PC1'],
                        y=pca_segment['PC2'],
                        z=pca_segment['PC3'],
                        mode='markers',
                        name=segment_name,
                        marker=dict(
                            size=5,
                            color=color_map.get(segment_id, '#023535'),
                            opacity=0.7,
                            line=dict(width=0.5, color='white')
                        ),
                        text=[f"LTV: ${ltv:,.0f}" for ltv in segment_data['total_spend']],
                        hovertemplate=(
                            f'<b>{segment_name}</b><br>'
                            'PC1: %{x:.2f}<br>'
                            'PC2: %{y:.2f}<br>'
                            'PC3: %{z:.2f}<br>'
                            '%{text}<extra></extra>'
                        )
                    ),
                    row=1, col=1
                )
            
            # Update 3D scene
            fig.update_scenes(
                dict(
                    xaxis_title="PC1 (Spend & Frequency)",
                    yaxis_title="PC2 (Service Mix)",
                    zaxis_title="PC3 (Recency & Tenure)",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
                ),
                row=1, col=1
            )
        
        # 2. Segment Size & Avg Value - Clear dual-axis visualization
        segment_sorted = self.segment_profiles.sort_values('total_revenue', ascending=False)
        
        # Customer count bars
        fig.add_trace(
            go.Bar(
                x=segment_sorted['name'],
                y=segment_sorted['size'],
                name='# of Customers',
                marker_color=[color_map.get(sid, '#023535') for sid in segment_sorted['segment_id']],
                text=[f"{int(s)}" for s in segment_sorted['size']],
                textposition='outside',
                textfont=dict(size=10),
                hovertemplate='<b>%{x}</b><br>Customers: %{y:,.0f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Average LTV line on secondary y-axis  
        fig.add_trace(
            go.Scatter(
                x=segment_sorted['name'],
                y=segment_sorted['avg_total_spend'],
                name='Avg Customer LTV',
                mode='lines+markers+text',
                line=dict(color='#ff6b6b', width=3),
                marker=dict(size=12, symbol='diamond', line=dict(width=2, color='white')),
                text=[f"${v/1000:.1f}K" for v in segment_sorted['avg_total_spend']],
                textposition='top center',
                textfont=dict(size=9, color='#ff6b6b'),
                yaxis='y2',
                hovertemplate='<b>%{x}</b><br>Avg LTV: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Service Mix by Segment - Stacked bars
        # DEBUG: Show what columns we have
        import streamlit as st
        service_cols = [col for col in self.segment_profiles.columns if col.startswith('pct_')]
        
        st.write("### DEBUG INFO")
        st.write(f"**Total columns in segment_profiles:** {len(self.segment_profiles.columns)}")
        st.write(f"**Columns starting with pct_:** {service_cols}")
        st.write(f"**Number of pct_ columns:** {len(service_cols)}")
        st.write(f"**Segment names:** {self.segment_profiles['name'].tolist()}")
        
        # Show the actual data
        st.write("**Segment profiles data:**")
        st.dataframe(self.segment_profiles)
        
        service_names = [col.replace('pct_', '').title() for col in service_cols]
        
        colors_services = {
            'Installation': '#00b894',
            'Cooling': '#008f8c',
            'Heating': '#ff6b6b',
            'Maintenance': '#23606e',
            'Emergency': '#667eea'
        }
        
        for idx, service_col in enumerate(service_cols):
            service_name = service_names[idx]
            
            fig.add_trace(
                go.Bar(
                    x=self.segment_profiles['name'],
                    y=self.segment_profiles[service_col],
                    name=service_name,
                    marker_color=colors_services.get(service_name, '#023535'),
                    text=[f"{v:.0f}%" for v in self.segment_profiles[service_col]],
                    textposition='inside',
                    hovertemplate=f'<b>{service_name}</b><br>%{{x}}<br>%{{y:.1f}}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. RFM Profile by Segment - Clearer metric labels
        # Normalize RFM metrics for comparison (0-100 scale)
        max_recency = self.segment_profiles['avg_recency'].max()
        max_freq = self.segment_profiles['avg_frequency'].max()
        max_monetary = self.segment_profiles['avg_total_spend'].max()
        
        # Invert recency (lower days = better = higher score)
        self.segment_profiles['recency_score'] = 100 * (1 - self.segment_profiles['avg_recency'] / max_recency)
        self.segment_profiles['frequency_score'] = 100 * self.segment_profiles['avg_frequency'] / max_freq
        self.segment_profiles['monetary_score'] = 100 * self.segment_profiles['avg_total_spend'] / max_monetary
        
        # Plot each RFM dimension with intuitive labels
        fig.add_trace(
            go.Scatter(
                x=self.segment_profiles['name'],
                y=self.segment_profiles['recency_score'],
                mode='lines+markers',
                name='Recency (How Recently)',
                line=dict(color='#00b894', width=3),
                marker=dict(size=10, line=dict(width=2, color='white')),
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    '<b>Recency:</b> %{y:.0f}/100<br>'
                    'Last service: %{customdata:.0f} days ago<br>'
                    '<i>(Lower = more recent = better)</i>'
                    '<extra></extra>'
                ),
                customdata=self.segment_profiles['avg_recency']
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.segment_profiles['name'],
                y=self.segment_profiles['frequency_score'],
                mode='lines+markers',
                name='Frequency (How Often)',
                line=dict(color='#008f8c', width=3),
                marker=dict(size=10, line=dict(width=2, color='white')),
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    '<b>Frequency:</b> %{y:.0f}/100<br>'
                    'Avg transactions: %{customdata:.1f}<br>'
                    '<i>(More visits = higher score)</i>'
                    '<extra></extra>'
                ),
                customdata=self.segment_profiles['avg_frequency']
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.segment_profiles['name'],
                y=self.segment_profiles['monetary_score'],
                mode='lines+markers',
                name='Monetary (How Much)',
                line=dict(color='#ff6b6b', width=3),
                marker=dict(size=10, line=dict(width=2, color='white')),
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    '<b>Monetary:</b> %{y:.0f}/100<br>'
                    'Avg lifetime value: $%{customdata:,.0f}<br>'
                    '<i>(More spend = higher score)</i>'
                    '<extra></extra>'
                ),
                customdata=self.segment_profiles['avg_total_spend']
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=950,
            showlegend=True,
            title_text="",
            title_x=0.5,
            title_font=dict(size=18, color='#023535'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            barmode='stack',
            margin=dict(l=80, r=80, t=120, b=80)
        )
        
        # Update axes with clear labels
        fig.update_xaxes(title_text="Customer Segment", row=1, col=2, tickangle=-45)
        fig.update_yaxes(title_text="Number of Customers", row=1, col=2, secondary_y=False)
        fig.update_yaxes(title_text="Average LTV ($)", row=1, col=2, secondary_y=True)
        
        fig.update_xaxes(title_text="Customer Segment", row=2, col=1, tickangle=-45)
        fig.update_yaxes(title_text="Service Mix (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="Customer Segment", row=2, col=2, tickangle=-45)
        fig.update_yaxes(title_text="RFM Score (0-100, higher is better)", row=2, col=2, range=[0, 105])
        
        # Update 3D scene
        scene_dict = dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        )
        fig.update_scenes(scene_dict)
        
        return fig
    
    def get_insights(self) -> list:
        """Generate data-driven insights from segmentation."""
        if self.customer_features is None:
            self.load_data()
        
        insights = []
        
        insights.append(
            f"3D clustering reveals {self.n_clusters} distinct customer segments from {len(self.customer_features)} customers "
            f"- clear separation visible in PCA space"
        )
        
        if self.segment_profiles is not None and len(self.segment_profiles) > 0:
            # Highest revenue segment
            highest_revenue = self.segment_profiles.nlargest(1, 'total_revenue').iloc[0]
            revenue_pct = (highest_revenue['total_revenue'] / self.segment_profiles['total_revenue'].sum()) * 100
            
            insights.append(
                f"{highest_revenue['name']} drives {revenue_pct:.0f}% of total revenue "
                f"({highest_revenue['size']} customers, ${highest_revenue['avg_total_spend']:,.0f} avg LTV)"
            )
            
            # Most valuable customers (RFM profile)
            highest_ltv = self.segment_profiles.nlargest(1, 'avg_total_spend').iloc[0]
            if highest_ltv['segment_id'] != highest_revenue['segment_id']:
                insights.append(
                    f"{highest_ltv['name']} has highest average customer value "
                    f"(${highest_ltv['avg_total_spend']:,.0f} LTV) - RFM profile shows high monetary score"
                )
            
            # Service mix insight
            installation_heavy = self.segment_profiles[
                self.segment_profiles.get('pct_installation', pd.Series([0])) > 40
            ]
            if len(installation_heavy) > 0:
                insights.append(
                    f"Service mix analysis: {installation_heavy.iloc[0]['name']} are {installation_heavy.iloc[0]['pct_installation']:.0f}% installation-focused - "
                    f"cross-sell maintenance contracts for recurring revenue"
                )
            
            # At-risk customers
            at_risk = self.segment_profiles[self.segment_profiles['avg_recency'] > 365]
            if len(at_risk) > 0:
                at_risk_total = at_risk['size'].sum()
                at_risk_pct = (at_risk_total / len(self.customer_features)) * 100
                insights.append(
                    f"{at_risk_total} customers ({at_risk_pct:.0f}%) haven't purchased in 1+ years - "
                    f"'{at_risk.iloc[0]['name']}' segment needs win-back campaign (low recency score)"
                )
            
            # RFM profile insight
            high_freq = self.segment_profiles.nlargest(1, 'avg_frequency').iloc[0]
            if high_freq['avg_frequency'] >= 3:
                insights.append(
                    f"RFM analysis: '{high_freq['name']}' averages {high_freq['avg_frequency']:.1f} transactions - "
                    f"highest frequency score indicates strong engagement"
                )
        
        # PCA insight
        if self.pca is not None:
            cum_var = self.pca.explained_variance_ratio_.cumsum()
            if len(cum_var) >= 3:
                insights.append(
                    f"First 3 PCA components explain {cum_var[2]*100:.0f}% of customer variation - "
                    f"dimensionality reduction effective for visualization"
                )
        
        # Connection to other analyses
        insights.append(
            "Connection to Churn Prediction: Use segment characteristics (recency, frequency, service mix) "
            "to predict which customers in each group are most at risk"
        )
        
        return insights
    
    # Backward compatibility
    @property
    def insights(self) -> list:
        return self.get_insights()
    
    def get_recommendations(self) -> list:
        """Generate actionable recommendations."""
        recommendations = []
        
        if self.segment_profiles is not None:
            # Target high-value segments
            highest_ltv = self.segment_profiles.nlargest(1, 'avg_total_spend').iloc[0]
            recommendations.append(
                f"VIP treatment for {highest_ltv['name']}: Priority scheduling, dedicated technician, "
                f"annual check-ins - these customers are worth ${highest_ltv['avg_total_spend']:,.0f} each"
            )
            
            # Win back dormant
            at_risk = self.segment_profiles[self.segment_profiles['avg_recency'] > 365]
            if len(at_risk) > 0:
                recommendations.append(
                    f"Win-back campaign: Email {at_risk.iloc[0]['name']} with 'We Miss You' offer - "
                    f"seasonal tune-up discount or free safety inspection"
                )
            
            # Convert to maintenance
            non_maintenance = self.segment_profiles[
                self.segment_profiles.get('pct_maintenance', pd.Series([0])) < 20
            ]
            if len(non_maintenance) > 0:
                recommendations.append(
                    f"**Convert to maintenance contracts**: After every repair, offer "
                    f"{non_maintenance.iloc[0]['name']} a maintenance plan - prevents future emergencies"
                )
            
            # Protect installation clients
            installation_seg = self.segment_profiles[
                self.segment_profiles.get('pct_installation', pd.Series([0])) > 40
            ]
            if len(installation_seg) > 0:
                recommendations.append(
                    f"**Installation client retention**: {installation_seg.iloc[0]['name']} made big investments - "
                    f"offer extended warranties and first-year maintenance included"
                )
        
        recommendations.append(
            "Create segment-specific email campaigns: maintenance customers get seasonal reminders, "
            "installation customers get warranty info, at-risk get win-back offers"
        )
        
        recommendations.append(
            "Next step: Use Pricing Analysis to ensure each segment has appropriate service packages "
            "and price points matching their willingness to pay"
        )
        
        return recommendations
    
    # Backward compatibility
    @property
    def recommendations(self) -> list:
        return self.get_recommendations()
    
    @property
    def business_impact(self) -> str:
        return "RFM-based segmentation can increase marketing ROI by 3-5x through targeted campaigns, improve retention by 25% through segment-specific strategies, and identify high-LTV customers worth 10x more than average"
