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
            '**Source**: ServiceTitan (Field Service Software)',
            '**Dataset**: customer_segmentation_transactions.csv',
            '**Records**: 1,778 service transactions, 178 unique customers',
            '**Contains**: Customer ID, date, service type/category, amount, payment method, technician, duration, parts/labor costs, follow-up needs'
        ]
    
    # Backward compatibility
    @property
    def data_inputs(self) -> list:
        return self.data_collected
    
    @property
    def methodology(self) -> str:
        return """We use the following analytical techniques to help Ron understand his different customer types and tailor his approach to each group:

**RFM Analysis (Recency, Frequency, Monetary)** - The gold standard for customer segmentation. How recently did they buy? How often? How much did they spend? These three metrics reveal customer value and engagement.

**K-Means Clustering** - A machine learning algorithm that automatically groups customers with similar behaviors together, creating natural segments without manual rules.

**Principal Component Analysis (PCA)** - Reduces complex customer data into 3D visualizations showing how different segments naturally separate.

**Customer Lifetime Value (LTV) calculation** - Estimates how much each customer is worth over their entire relationship with Ron's business.

**Why this works for Ron:** Instead of treating all customers the same, Ron can create targeted marketing, pricing, and service strategies for each segment (VIPs get white-glove service, maintenance customers get contract renewals, etc.).

**If results aren't strong enough, we could:**
- Try different clustering algorithms (DBSCAN, Hierarchical) to find better segment boundaries
- Add behavioral features like service type preferences or response to promotions
- Use predictive segmentation to see how customers move between segments over time
- Combine with geographic or demographic data for richer profiles"""
    
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
        
        # Add service mix features
        service_mix = df.groupby("customer_id")["service_category"].value_counts().unstack(fill_value=0)
        service_mix = service_mix.div(service_mix.sum(axis=1), axis=0) * 100  # Convert to percentages
        
        # Ensure all categories exist
        for cat in ['Installation', 'Cooling', 'Heating', 'Maintenance']:
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
    
    def create_segment_profiles(self, customer_features, labels):
        """Create detailed profiles for each segment with descriptive names."""
        customer_features_copy = customer_features.copy()
        customer_features_copy['segment'] = labels
        
        profiles = []
        for segment_id in sorted(set(labels)):
            segment_data = customer_features_copy[customer_features_copy['segment'] == segment_id]
            
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
            
            # Add service mix percentages
            for col in customer_features.columns:
                if col.startswith('pct_'):
                    profile[col] = segment_data[col].mean()
            
            profile['avg_tenure'] = segment_data.get('customer_tenure_days', pd.Series([0])).mean()
            
            # Assign descriptive names based on characteristics
            if profile['avg_total_spend'] > customer_features['total_spend'].quantile(0.75):
                if profile.get('pct_installation', 0) > 40:
                    profile['name'] = 'VIP Installation Clients'
                else:
                    profile['name'] = 'High-Value Regulars'
            elif profile['avg_frequency'] >= 3 and profile.get('pct_maintenance', 0) > 30:
                profile['name'] = 'Maintenance Contract Holders'
            elif profile['avg_recency'] < 90:
                profile['name'] = 'Recent Customers'
            elif profile['avg_recency'] > 365:
                profile['name'] = 'At-Risk / Dormant'
            else:
                if profile.get('pct_cooling', 0) + profile.get('pct_heating', 0) > 60:
                    profile['name'] = 'Repair-Focused'
                else:
                    profile['name'] = 'Occasional Service'
            
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
        self.segment_profiles = self.create_segment_profiles(self.customer_features, self.labels)
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
                'Segment Characteristics & Profile'
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
        
        # 2. Segment Size & Avg Value
        segment_sorted = self.segment_profiles.sort_values('total_revenue', ascending=False)
        
        fig.add_trace(
            go.Bar(
                x=segment_sorted['name'],
                y=segment_sorted['size'],
                name='Customer Count',
                marker_color=[color_map.get(sid, '#023535') for sid in segment_sorted['segment_id']],
                text=segment_sorted['size'],
                textposition='outside',
                yaxis='y2',
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Customers: %{y}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Add average spend as line on secondary axis
        fig.add_trace(
            go.Scatter(
                x=segment_sorted['name'],
                y=segment_sorted['avg_total_spend'],
                name='Avg LTV',
                mode='lines+markers',
                line=dict(color='#ff6b6b', width=3),
                marker=dict(size=10, symbol='diamond'),
                yaxis='y2',
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Avg LTV: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Service Mix by Segment - Stacked bars
        service_cols = [col for col in self.segment_profiles.columns if col.startswith('pct_')]
        service_names = [col.replace('pct_', '').title() for col in service_cols]
        
        colors_services = {
            'Installation': '#00b894',
            'Cooling': '#008f8c',
            'Heating': '#ff6b6b',
            'Maintenance': '#23606e'
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
        
        # 4. Segment Descriptions - Key Characteristics
        # Instead of a chart, we'll create a text-based summary using annotations
        # First, hide the subplot axes
        fig.update_xaxes(visible=False, row=2, col=2)
        fig.update_yaxes(visible=False, row=2, col=2)
        
        # Create segment characteristic descriptions
        segment_descriptions = {
            'VIP Installation Clients': {
                'icon': '⭐',
                'spend': 'High',
                'frequency': 'Low-Med',
                'characteristics': 'Major purchases • Big-ticket items • Price insensitive',
                'color': '#00b894'
            },
            'High-Value Regulars': {
                'icon': '💎',
                'spend': 'High',
                'frequency': 'High',
                'characteristics': 'Loyal customers • Multiple services • Best LTV',
                'color': '#008f8c'
            },
            'Maintenance Contract Holders': {
                'icon': '🔧',
                'spend': 'Medium',
                'frequency': 'Regular',
                'characteristics': 'Predictable revenue • Seasonal tune-ups • Contract-based',
                'color': '#23606e'
            },
            'Occasional Service': {
                'icon': '📅',
                'spend': 'Low-Med',
                'frequency': 'Low',
                'characteristics': 'Sporadic calls • Repair-focused • Price sensitive',
                'color': '#ffa07a'
            },
            'At-Risk / Dormant': {
                'icon': '⚠️',
                'spend': 'Varies',
                'frequency': 'Very Low',
                'characteristics': 'Inactive 12+ months • Churn risk • Win-back target',
                'color': '#ff6b6b'
            },
            'Recent Customers': {
                'icon': '🆕',
                'spend': 'Varies',
                'frequency': 'New',
                'characteristics': 'First 90 days • Growth opportunity • Build loyalty',
                'color': '#98d8c8'
            },
            'Repair-Focused': {
                'icon': '🔨',
                'spend': 'Medium',
                'frequency': 'Medium',
                'characteristics': 'Emergency calls • Reactive service • Seasonal peaks',
                'color': '#667eea'
            }
        }
        
        # Add descriptions as annotations for each segment we actually have
        y_position = 0.95
        for _, segment in self.segment_profiles.iterrows():
            seg_name = segment['name']
            if seg_name in segment_descriptions:
                desc = segment_descriptions[seg_name]
                
                # Add segment description as annotation
                fig.add_annotation(
                    xref="x4", yref="y4",
                    x=0.5, y=y_position,
                    text=(
                        f"<b>{desc['icon']} {seg_name}</b><br>"
                        f"<i>{segment['size']} customers | ${segment['avg_total_spend']:,.0f} avg LTV</i><br>"
                        f"<span style='font-size:10px'>{desc['characteristics']}</span>"
                    ),
                    showarrow=False,
                    align="left",
                    xanchor="left",
                    font=dict(size=11, color=desc['color']),
                    bordercolor=desc['color'],
                    borderwidth=2,
                    borderpad=8,
                    bgcolor="white",
                    opacity=0.95
                )
                y_position -= 0.22
        
        # Update layout
        fig.update_layout(
            height=950,
            showlegend=True,
            title_text="Customer Segmentation: Understanding Ron's Customer Base",
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
        
        # Update axes
        fig.update_xaxes(title_text="Customer Segment", row=1, col=2, tickangle=-45)
        fig.update_yaxes(title_text="Number of Customers", row=1, col=2, secondary_y=False, rangemode='tozero')
        
        fig.update_xaxes(title_text="Customer Segment", row=2, col=1, tickangle=-45)
        fig.update_yaxes(title_text="Service Mix (%)", row=2, col=1, range=[0, 105])
        
        fig.update_xaxes(title_text="Customer Segment", row=2, col=2, tickangle=-45)
        fig.update_yaxes(title_text="RFM Score (0-100)", row=2, col=2, range=[0, 108])
        
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
            f"**3D clustering reveals {self.n_clusters} distinct customer segments** from {len(self.customer_features)} customers "
            f"- clear separation visible in PCA space"
        )
        
        if self.segment_profiles is not None and len(self.segment_profiles) > 0:
            # Highest revenue segment
            highest_revenue = self.segment_profiles.nlargest(1, 'total_revenue').iloc[0]
            revenue_pct = (highest_revenue['total_revenue'] / self.segment_profiles['total_revenue'].sum()) * 100
            
            insights.append(
                f"**{highest_revenue['name']}** drives {revenue_pct:.0f}% of total revenue "
                f"({highest_revenue['size']} customers, ${highest_revenue['avg_total_spend']:,.0f} avg LTV)"
            )
            
            # Most valuable customers (RFM profile)
            highest_ltv = self.segment_profiles.nlargest(1, 'avg_total_spend').iloc[0]
            if highest_ltv['segment_id'] != highest_revenue['segment_id']:
                insights.append(
                    f"**{highest_ltv['name']}** has highest average customer value "
                    f"(${highest_ltv['avg_total_spend']:,.0f} LTV) - RFM profile shows high monetary score"
                )
            
            # Service mix insight
            installation_heavy = self.segment_profiles[
                self.segment_profiles.get('pct_installation', pd.Series([0])) > 40
            ]
            if len(installation_heavy) > 0:
                insights.append(
                    f"**Service mix analysis**: {installation_heavy.iloc[0]['name']} are {installation_heavy.iloc[0]['pct_installation']:.0f}% installation-focused - "
                    f"cross-sell maintenance contracts for recurring revenue"
                )
            
            # At-risk customers
            at_risk = self.segment_profiles[self.segment_profiles['avg_recency'] > 365]
            if len(at_risk) > 0:
                at_risk_total = at_risk['size'].sum()
                at_risk_pct = (at_risk_total / len(self.customer_features)) * 100
                insights.append(
                    f"**{at_risk_total} customers ({at_risk_pct:.0f}%) haven't purchased in 1+ years** - "
                    f"'{at_risk.iloc[0]['name']}' segment needs win-back campaign (low recency score)"
                )
            
            # RFM profile insight
            high_freq = self.segment_profiles.nlargest(1, 'avg_frequency').iloc[0]
            if high_freq['avg_frequency'] >= 3:
                insights.append(
                    f"**RFM analysis**: '{high_freq['name']}' averages {high_freq['avg_frequency']:.1f} transactions - "
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
            "**Connection to Churn Prediction**: Use segment characteristics (recency, frequency, service mix) "
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
                f"**VIP treatment for {highest_ltv['name']}**: Priority scheduling, dedicated technician, "
                f"annual check-ins - these customers are worth ${highest_ltv['avg_total_spend']:,.0f} each"
            )
            
            # Win back dormant
            at_risk = self.segment_profiles[self.segment_profiles['avg_recency'] > 365]
            if len(at_risk) > 0:
                recommendations.append(
                    f"**Win-back campaign**: Email {at_risk.iloc[0]['name']} with 'We Miss You' offer - "
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
            "**Next step**: Use Pricing Analysis to ensure each segment has appropriate service packages "
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
