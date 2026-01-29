"""Marketing Impact Analysis Module - Amralytics Methodology"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .base_analysis import BaseAnalysis


class MarketingImpact(BaseAnalysis):
    """Marketing channel performance, ROI, and customer journey analysis"""
    
    def __init__(self):
        self.marketing_df = None
        self.channel_performance = None
        self.lagged_impact = None
        self.funnel_data = None
    
    @property
    def icon(self) -> str:
        return '📱'
    
    @property
    def color(self) -> str:
        return '#23606e'
    
    @property
    def rons_challenge(self) -> str:
        return """Ron has been spending money on various marketing channels: Google Ads, Instagram, 
        LinkedIn, Yelp, and even some print advertising. He's also got a referral program.
        
**But which channels actually work?** Ron needs to know where to invest his limited marketing 
budget to get the best return. And how long does it take from ad click to actual service call?"""
    
    # Backward compatibility
    @property
    def business_question(self) -> str:
        return self.rons_challenge
    
    @property
    def data_collected(self) -> list:
        return [
            'Campaign impressions and clicks - **Google Analytics, Meta Ads Manager**',
            'Lead generation by channel - **Google Analytics, Meta Ads Manager**',
            'Conversion rates - **ServiceTitan, Google Analytics**',
            'Marketing spend by channel - **Google Ads, Meta Ads Manager, Print vendors**',
            'Revenue attribution - **ServiceTitan, Google Analytics**',
            'ROAS calculations - **QuickBooks, Google Analytics**'
    ]
    
    # Backward compatibility
    @property
    def data_inputs(self) -> list:
        return self.data_collected
    
    @property
    def methodology(self) -> str:
        return """We use the following analytical techniques to determine which marketing channels are worth Ron's money:

**ROAS Analysis (Return on Ad Spend)** - For every dollar Ron spends on each channel (Google Ads, Facebook, mailers), how many dollars of revenue does he get back? Anything under 3x is concerning.

**Full funnel metrics** - Tracking each channel from impressions → clicks → leads → conversions → revenue to find where the drop-off happens.

**Conversion lag analysis** - How long between seeing an ad and making a purchase? Emergency services convert in hours, installations might take weeks.

**Cost Per Acquisition (CPA)** - What does it cost Ron to acquire one customer through each channel? Combined with LTV tells him which channels are profitable long-term.

**Why this works for Ron:** Immediately shows which channels are wasting money (cut them) and which are winners (double down). Most small businesses guess at this - Ron will know.

**If results aren't strong enough, we could:**
- Add attribution modeling (multi-touch - customer saw 3 ads before converting, who gets credit?)
- Include customer quality by channel (some channels bring high-LTV customers, others bring one-timers)
- Build predictive models for optimal budget allocation across channels
- Test incrementality (what would happen if we turned off this channel entirely?)"""
    # Backward compatibility
    @property
    def technical_output(self) -> str:
        return self.methodology
    
    @property
    def data_file(self) -> str:
        return 'marketing_data.csv'
    
    def load_data(self, filepath: str = None):
        """Load and process marketing data."""
        if filepath is None:
            filepath = f'data/{self.data_file}'
        
        self.marketing_df = pd.read_csv(filepath)
        self.marketing_df['date'] = pd.to_datetime(self.marketing_df['date'])
        
        # Calculate channel performance
        self._calculate_channel_performance()
        
        # Analyze lag effects
        self._analyze_lag_effects()
        
        # Build funnel data
        self._build_funnel()
        
        return self.marketing_df
    
    def _calculate_channel_performance(self):
        """Calculate aggregate performance by channel."""
        self.channel_performance = self.marketing_df.groupby('channel').agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'leads': 'sum',
            'conversions': 'sum',
            'revenue': 'sum'
        }).reset_index()
        
        # Calculate metrics
        self.channel_performance['roas'] = self.channel_performance['revenue'] / self.channel_performance['spend']
        self.channel_performance['cpa'] = self.channel_performance['spend'] / self.channel_performance['conversions']
        self.channel_performance['conversion_rate'] = (
            self.channel_performance['conversions'] / self.channel_performance['leads'] * 100
        )
        self.channel_performance['ctr'] = (
            self.channel_performance['clicks'] / self.channel_performance['impressions'] * 100
        )
        
        # Sort by ROAS descending
        self.channel_performance = self.channel_performance.sort_values('roas', ascending=False)
    
    def _analyze_lag_effects(self):
        """Analyze conversion timing by service type."""
        # Simulate lag data (in real implementation, this would come from actual conversion tracking)
        # For now, create realistic lag patterns based on service type
        
        lag_patterns = {
            'Emergency': {'0-3 days': 0.65, '4-7 days': 0.25, '8-14 days': 0.08, '15+ days': 0.02},
            'Cooling': {'0-3 days': 0.35, '4-7 days': 0.40, '8-14 days': 0.20, '15+ days': 0.05},
            'Heating': {'0-3 days': 0.30, '4-7 days': 0.45, '8-14 days': 0.20, '15+ days': 0.05},
            'Maintenance': {'0-3 days': 0.15, '4-7 days': 0.30, '8-14 days': 0.35, '15+ days': 0.20},
            'Installation': {'0-3 days': 0.10, '4-7 days': 0.25, '8-14 days': 0.40, '15+ days': 0.25},
            'Mixed': {'0-3 days': 0.25, '4-7 days': 0.35, '8-14 days': 0.30, '15+ days': 0.10}
        }
        
        lag_data = []
        for service_type, distribution in lag_patterns.items():
            for lag_period, percentage in distribution.items():
                lag_data.append({
                    'service_type': service_type,
                    'lag_period': lag_period,
                    'percentage': percentage * 100
                })
        
        self.lagged_impact = pd.DataFrame(lag_data)
    
    def _build_funnel(self):
        """Build marketing funnel data."""
        # Sum only numeric columns
        numeric_cols = ['impressions', 'clicks', 'leads', 'conversions', 'revenue']
        totals = self.marketing_df[numeric_cols].sum()
        
        self.funnel_data = pd.DataFrame({
            'stage': ['Impressions', 'Clicks', 'Leads', 'Conversions', 'Revenue ($)'],
            'value': [
                totals['impressions'],
                totals['clicks'],
                totals['leads'],
                totals['conversions'],
                totals['revenue'] / 100  # Scale down for visualization
            ],
            'percentage': [
                100.0,
                (totals['clicks'] / totals['impressions'] * 100),
                (totals['leads'] / totals['impressions'] * 100),
                (totals['conversions'] / totals['impressions'] * 100),
                (totals['revenue'] / totals['impressions'] * 100)
            ]
        })
    
    def create_visualization(self):
        """Create 4-panel marketing impact dashboard."""
        if self.marketing_df is None:
            self.load_data()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Channel Performance: ROAS vs Spend',
                'Conversion Lag by Service Type',
                'Marketing Funnel: Impressions to Revenue',
                'Channel Comparison: Key Metrics'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "funnel"}, {"type": "bar"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # 1. Channel Performance - Bubble chart (ROAS vs Spend, size = Revenue)
        colors_map = {
            'Google Ads': '#008f8c',
            'Referral Program': '#00b894',
            'LinkedIn': '#23606e',
            'Instagram': '#ff6b6b',
            'Yelp': '#ffa07a',
            'Print': '#ff8c8c'
        }
        
        for _, row in self.channel_performance.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row['spend']],
                    y=[row['roas']],
                    mode='markers+text',
                    name=row['channel'],
                    marker=dict(
                        size=row['revenue'] / 500,  # Scale for visualization
                        color=colors_map.get(row['channel'], '#023535'),
                        line=dict(width=2, color='white')
                    ),
                    text=row['channel'],
                    textposition='top center',
                    hovertemplate=(
                        f"<b>{row['channel']}</b><br>"
                        f"Spend: ${row['spend']:,.0f}<br>"
                        f"ROAS: {row['roas']:.1f}x<br>"
                        f"Revenue: ${row['revenue']:,.0f}<br>"
                        f"Conversions: {row['conversions']:.0f}<extra></extra>"
                    )
                ),
                row=1, col=1
            )
        
        # Add break-even line
        max_spend = self.channel_performance['spend'].max()
        fig.add_trace(
            go.Scatter(
                x=[0, max_spend],
                y=[1, 1],
                mode='lines',
                name='Break-even (1x)',
                line=dict(dash='dash', color='#023535', width=2),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # 2. Conversion Lag by Service Type - Stacked bar
        service_types = self.lagged_impact['service_type'].unique()
        lag_periods = ['0-3 days', '4-7 days', '8-14 days', '15+ days']
        
        colors_lag = ['#00b894', '#008f8c', '#23606e', '#015958']
        
        for idx, period in enumerate(lag_periods):
            period_data = self.lagged_impact[self.lagged_impact['lag_period'] == period]
            
            fig.add_trace(
                go.Bar(
                    x=period_data['service_type'],
                    y=period_data['percentage'],
                    name=period,
                    marker_color=colors_lag[idx],
                    text=[f"{v:.0f}%" for v in period_data['percentage']],
                    textposition='inside'
                ),
                row=1, col=2
            )
        
        # 3. Marketing Funnel
        fig.add_trace(
            go.Funnel(
                y=self.funnel_data['stage'],
                x=self.funnel_data['value'],
                textposition='inside',
                textinfo='value+percent initial',
                marker=dict(
                    color=['#008f8c', '#23606e', '#015958', '#023535', '#00b894'],
                ),
                connector=dict(line=dict(color='#fffcf2', width=2)),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Count: %{x:,.0f}<br>"
                    "% of Impressions: %{percentInitial}<extra></extra>"
                )
            ),
            row=2, col=1
        )
        
        # 4. Channel Comparison - Grouped bars for CPA and Conversion Rate
        fig.add_trace(
            go.Bar(
                x=self.channel_performance['channel'],
                y=self.channel_performance['cpa'],
                name='CPA ($)',
                marker_color='#ff6b6b',
                text=[f"${v:.0f}" for v in self.channel_performance['cpa']],
                textposition='outside',
                yaxis='y4'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=self.channel_performance['channel'],
                y=self.channel_performance['conversion_rate'],
                name='Conversion Rate (%)',
                marker_color='#008f8c',
                text=[f"{v:.1f}%" for v in self.channel_performance['conversion_rate']],
                textposition='outside',
                yaxis='y4'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1050,
            showlegend=True,
            title_text="Marketing Impact: Channel Performance & Customer Journey",
            title_x=0.5,
            title_font=dict(size=18, color='#023535'),
            barmode='group'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Marketing Spend ($)", row=1, col=1)
        fig.update_yaxes(title_text="ROAS (Return on Ad Spend)", row=1, col=1)
        
        fig.update_xaxes(title_text="Service Type", row=1, col=2, tickangle=-45)
        fig.update_yaxes(title_text="% of Conversions", row=1, col=2)
        
        fig.update_xaxes(title_text="Channel", row=2, col=2, tickangle=-45)
        fig.update_yaxes(title_text="Value", row=2, col=2)
        
        return fig
    
    def get_insights(self) -> list:
        """Generate data-driven insights from marketing analysis."""
        if self.marketing_df is None:
            self.load_data()
        
        insights = []
        
        # Top performing channel
        best_channel = self.channel_performance.iloc[0]
        worst_channel = self.channel_performance.iloc[-1]
        
        insights.append(
            f"Best channel: {best_channel['channel']} with {best_channel['roas']:.1f}x ROAS "
            f"(${best_channel['revenue']:,.0f} revenue from ${best_channel['spend']:,.0f} spend)"
        )
        
        insights.append(
            f"Worst channel: {worst_channel['channel']} with {worst_channel['roas']:.1f}x ROAS "
            f"- consider reducing or eliminating this channel"
        )
        
        # CPA comparison
        avg_cpa = self.channel_performance['cpa'].mean()
        efficient_channels = self.channel_performance[self.channel_performance['cpa'] < avg_cpa]
        
        if len(efficient_channels) > 0:
            insights.append(
                f"Channels with below-average CPA: {', '.join(efficient_channels['channel'].tolist())} "
                f"(avg CPA: ${avg_cpa:.0f})"
            )
        
        # Conversion lag insights
        emergency_fast = self.lagged_impact[
            (self.lagged_impact['service_type'] == 'Emergency') & 
            (self.lagged_impact['lag_period'] == '0-3 days')
        ]['percentage'].values[0]
        
        insights.append(
            f"Conversion timing matters: {emergency_fast:.0f}% of emergency service conversions "
            f"happen within 3 days - fast response to ads is critical"
        )
        
        # Budget allocation recommendation
        total_spend = self.channel_performance['spend'].sum()
        top_2_spend = self.channel_performance.head(2)['spend'].sum()
        top_2_pct = (top_2_spend / total_spend) * 100
        
        insights.append(
            f"Top 2 channels account for {top_2_pct:.0f}% of spend - "
            f"consider reallocating budget from underperforming channels"
        )
        
        # Connection to other analyses
        insights.append(
            "Connection to Customer Segmentation: Different channels attract different customer types - "
            "cross-reference channel data with customer LTV from segmentation analysis"
        )
        
        return insights
    
    # Backward compatibility
    @property
    def insights(self) -> list:
        return self.get_insights()
    
    def get_recommendations(self) -> list:
        """Generate actionable recommendations."""
        if self.marketing_df is None:
            self.load_data()
        
        recommendations = []
        
        # Channel-specific recommendations
        best = self.channel_performance.iloc[0]['channel']
        worst = self.channel_performance.iloc[-1]['channel']
        
        recommendations.append(f"Increase investment in {best} - highest ROAS and best performer")
        recommendations.append(f"Stop or drastically reduce {worst} spend - poor ROI")
        
        # Emergency service timing
        recommendations.append(
            "For emergency service campaigns, ensure 24/7 phone coverage - "
            "65% of conversions happen within 3 days of ad click"
        )
        
        # Referral program
        if 'Referral Program' in self.channel_performance['channel'].values:
            referral_row = self.channel_performance[
                self.channel_performance['channel'] == 'Referral Program'
            ].iloc[0]
            if referral_row['roas'] > 10:
                recommendations.append(
                    "Expand referral program - excellent ROAS with minimal spend. "
                    "Consider increasing referral incentives to generate more volume"
                )
        
        # Service-specific targeting
        recommendations.append(
            "Target different ad copy by service type: emphasize urgency for emergency services, "
            "cost savings for maintenance, and quality for installations"
        )
        
        recommendations.append(
            "Next step: Use churn prediction to identify at-risk customers and retarget them "
            "through best-performing channels"
        )
        
        return recommendations
    
    # Backward compatibility
    @property
    def recommendations(self) -> list:
        return self.get_recommendations()
    
    @property
    def business_impact(self) -> str:
        return "Understanding which marketing channels deliver the best ROI allows Ron to reallocate budget from underperforming channels and scale what works, potentially doubling marketing effectiveness while reducing overall spend"
