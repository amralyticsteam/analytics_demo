"""Business Overview Module - Amralytics Methodology"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .base_analysis import BaseAnalysis


class BusinessOverview(BaseAnalysis):
    """High-level business overview with EDA and descriptive statistics"""
    
    def __init__(self):
        self.transactions_df = None
        self.monthly_summary = None
        self.service_summary = None
    
    @property
    def icon(self) -> str:
        return '🏢'
    
    @property
    def color(self) -> str:
        return '#23606e'
    
    @property
    def rons_challenge(self) -> str:
        return """Ron runs a successful HVAC business but has never analyzed his data systematically. 
He knows he's busy in summer (AC work) and winter (heating), but doesn't have clear metrics.
        
**What does Ron's business actually look like?** Revenue trends? Service mix? Customer patterns? 
Before diving into complex analyses, we need to understand the current state."""
    
    # Backward compatibility
    @property
    def business_question(self) -> str:
        return self.rons_challenge
    
    @property
    def data_collected(self) -> list:
        return [
        'Transaction history (1,778 service records, Oct 2023-Aug 2024) - **ServiceTitan**',
        'Service types and categories (Installation, Maintenance, Emergency, Cooling, Heating) - **ServiceTitan**',
        'Revenue breakdown by service category and time period - **QuickBooks**',
        'Customer payment methods (credit card, check, financing) and timing - **ServiceTitan**',
        'Technician assignments, service duration, parts and labor costs - **ServiceTitan**'
    ]
    
    # Backward compatibility
    @property
    def data_inputs(self) -> list:
        return self.data_collected
    
    @property
    def methodology(self) -> str:
        return """We use the following analytical techniques to give Ron a high-level view of business health:

**Exploratory Data Analysis (EDA)** - Looking at overall patterns in Ron's transaction data to spot trends, anomalies, and key metrics.

**Descriptive statistics** - Revenue totals, average transaction sizes, service volumes - the fundamental numbers that tell Ron if the business is growing or shrinking.

**Revenue trend analysis** - Month-over-month changes to identify growth patterns or concerning declines.

**Service mix analysis** - Which service categories (Installation, Maintenance, Emergency) drive the most revenue and how that's changing over time.

**Why this works for Ron:** This is the foundation - before diving into complex analyses, we need to understand the baseline health of the business.

**If results aren't clear enough, we could:**
- Add year-over-year comparisons to account for seasonality
- Include industry benchmarks to see how Ron compares to competitors
- Break down by technician or service area to spot performance differences
- Add customer acquisition cost (CAC) and lifetime value (LTV) metrics"""
    
    # Backward compatibility
    @property
    def technical_output(self) -> str:
        return self.methodology
    
    @property
    def data_file(self) -> str:
        return 'customer_segmentation_transactions.csv'
    
    def load_data(self, filepath: str = None):
        """Load and process transaction data."""
        if filepath is None:
            filepath = f'data/{self.data_file}'
        
        self.transactions_df = pd.read_csv(filepath)
        self.transactions_df['date'] = pd.to_datetime(self.transactions_df['date'])
        self.transactions_df = self.transactions_df.sort_values('date')
        
        # Create monthly aggregations using 'ME' (Month End) instead of deprecated 'M'
        self.transactions_df['year_month'] = self.transactions_df['date'].dt.to_period('M').dt.to_timestamp()
        
        # Monthly summary
        self.monthly_summary = self.transactions_df.groupby('year_month').agg({
            'amount': ['sum', 'mean', 'count'],
            'customer_id': 'nunique'
        }).reset_index()
        self.monthly_summary.columns = ['month', 'total_revenue', 'avg_ticket', 'transaction_count', 'unique_customers']
        
        # Service type summary (using service_category column)
        self.service_summary = self.transactions_df.groupby('service_category').agg({
            'amount': ['sum', 'mean', 'count'],
            'customer_id': 'nunique'
        }).reset_index()
        self.service_summary.columns = ['category', 'total_revenue', 'avg_ticket', 'job_count', 'unique_customers']
        self.service_summary = self.service_summary.sort_values('total_revenue', ascending=False)
        
        return self.transactions_df
    
    def create_visualization(self):
        """Create 4-panel business overview dashboard."""
        if self.transactions_df is None:
            self.load_data()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Monthly Revenue Trend',
                'Revenue by Service Category',
                'Average Transaction Size by Category',
                'Transaction Volume Over Time'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.22,
            horizontal_spacing=0.15
        )
        
        # 1. Monthly Revenue Trend
        fig.add_trace(
            go.Scatter(
                x=self.monthly_summary['month'],
                y=self.monthly_summary['total_revenue'],
                mode='lines+markers',
                name='Monthly Revenue',
                line=dict(color='#008f8c', width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(0, 143, 140, 0.2)',
                hovertemplate='%{x|%b %Y}<br>Revenue: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Revenue by Service Category
        colors_category = {
            'Heating': '#ff6b6b',
            'Cooling': '#008f8c',
            'Emergency': '#ffa07a',
            'Installation': '#00b894',
            'Maintenance': '#23606e'
        }
        
        category_colors = [colors_category.get(cat, '#023535') for cat in self.service_summary['category']]
        
        fig.add_trace(
            go.Bar(
                x=self.service_summary['category'],
                y=self.service_summary['total_revenue'],
                marker_color=category_colors,
                text=[f"${v/1000:.0f}K" for v in self.service_summary['total_revenue']],
                textposition='outside',
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Total Revenue: $%{y:,.0f}<br>Jobs: %{customdata}<extra></extra>',
                customdata=self.service_summary['job_count']
            ),
            row=1, col=2
        )
        
        # 3. Average Transaction Size
        fig.add_trace(
            go.Bar(
                x=self.service_summary['category'],
                y=self.service_summary['avg_ticket'],
                marker_color=category_colors,
                text=[f"${v:.0f}" for v in self.service_summary['avg_ticket']],
                textposition='outside',
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Avg Ticket: $%{y:.0f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Transaction Volume Over Time
        fig.add_trace(
            go.Scatter(
                x=self.monthly_summary['month'],
                y=self.monthly_summary['transaction_count'],
                mode='lines+markers',
                name='Transactions',
                line=dict(color='#23606e', width=3),
                marker=dict(size=8),
                hovertemplate='%{x|%b %Y}<br>Transactions: %{y}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Add unique customers as secondary line
        fig.add_trace(
            go.Scatter(
                x=self.monthly_summary['month'],
                y=self.monthly_summary['unique_customers'],
                mode='lines+markers',
                name='Unique Customers',
                line=dict(color='#ffa07a', width=2, dash='dash'),
                marker=dict(size=6),
                hovertemplate='%{x|%b %Y}<br>Customers: %{y}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text="Business Overview: Ron's HVAC Performance at a Glance",
            title_x=0.5,
            title_font=dict(size=18, color='#023535'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Month", row=1, col=1)
        fig.update_yaxes(title_text="Revenue ($)", row=1, col=1, rangemode='tozero')
        
        max_val_1_2 = self.service_summary['total_revenue'].max()
        fig.update_xaxes(title_text="Service Category", row=1, col=2, tickangle=-45)
        fig.update_yaxes(title_text="Total Revenue ($)", range=[0, max_val_1_2 * 1.15], row=1, col=2)
        
        fig.update_xaxes(title_text="Service Category", row=2, col=1, tickangle=-45)
        fig.update_yaxes(title_text="Average Transaction ($)", row=2, col=1, rangemode='tozero')
        
        fig.update_xaxes(title_text="Month", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2, rangemode='tozero')
        
        return fig
    
    def get_insights(self) -> list:
        """Generate data-driven insights from business overview."""
        if self.transactions_df is None:
            self.load_data()
        
        insights = []
        
        # Overall metrics
        total_revenue = self.transactions_df['amount'].sum()
        total_transactions = len(self.transactions_df)
        unique_customers = self.transactions_df['customer_id'].nunique()
        avg_ticket = self.transactions_df['amount'].mean()
        
        insights.append(
            f"Overall business: ${total_revenue:,.0f} total revenue from {total_transactions} transactions "
            f"across {unique_customers} customers (avg ticket: ${avg_ticket:.0f})"
        )
        
        # Top service category
        top_category = self.service_summary.iloc[0]
        top_pct = (top_category['total_revenue'] / total_revenue) * 100
        
        insights.append(
            f"Top revenue driver: {top_category['category']} accounts for {top_pct:.0f}% of revenue "
            f"(${top_category['total_revenue']:,.0f}) with {top_category['job_count']} jobs"
        )
        
        # Highest ticket size
        highest_ticket = self.service_summary.nlargest(1, 'avg_ticket').iloc[0]
        insights.append(
            f"Highest ticket size: {highest_ticket['category']} averages ${highest_ticket['avg_ticket']:.0f} per job"
        )
        
        # Revenue trend
        recent_3mo = self.monthly_summary.tail(3)['total_revenue'].mean()
        prior_3mo = self.monthly_summary.iloc[-6:-3]['total_revenue'].mean()
        trend_pct = ((recent_3mo - prior_3mo) / prior_3mo) * 100
        
        direction = "up" if trend_pct > 0 else "down"
        insights.append(
            f"Recent trend: Last 3 months average ${recent_3mo:,.0f}/month vs prior 3 months ${prior_3mo:,.0f}/month "
            f"({direction} {abs(trend_pct):.1f}%)"
        )
        
        # Customer concentration
        repeat_customers = self.transactions_df.groupby('customer_id').size()
        repeat_rate = (repeat_customers > 1).sum() / len(repeat_customers) * 100
        
        insights.append(
            f"Customer retention: {repeat_rate:.0f}% of customers have made multiple purchases - "
            f"{'strong retention' if repeat_rate > 50 else 'opportunity to improve loyalty'}"
        )
        
        # Payment method preference
        payment_dist = self.transactions_df['payment_method'].value_counts()
        top_payment = payment_dist.index[0]
        top_payment_pct = (payment_dist.iloc[0] / len(self.transactions_df)) * 100
        
        insights.append(
            f"Most common payment method: {top_payment} ({top_payment_pct:.0f}% of transactions)"
        )
        
        # Connection to other analyses
        insights.append(
            "Next step: Customer Segmentation will reveal distinct customer groups within this data, "
            "and Seasonality Analysis will explain the monthly revenue patterns"
        )
        
        return insights
    
    # Backward compatibility
    @property
    def insights(self) -> list:
        return self.get_insights()
    
    def get_recommendations(self) -> list:
        """Generate actionable recommendations."""
        if self.transactions_df is None:
            self.load_data()
        
        recommendations = []
        
        # Service focus
        top_category = self.service_summary.iloc[0]
        recommendations.append(
            f"Focus marketing and capacity on {top_category['category']} - it's the revenue engine. "
            f"Ensure adequate parts inventory and trained staff"
        )
        
        # Ticket size optimization
        lowest_ticket = self.service_summary.nsmallest(1, 'avg_ticket').iloc[0]
        if lowest_ticket['avg_ticket'] < 200:
            recommendations.append(
                f"{lowest_ticket['category']} has low average ticket (${lowest_ticket['avg_ticket']:.0f}) - "
                f"explore bundling opportunities to increase transaction value"
            )
        
        # Revenue consistency
        revenue_std = self.monthly_summary['total_revenue'].std()
        revenue_mean = self.monthly_summary['total_revenue'].mean()
        cv = (revenue_std / revenue_mean) * 100
        
        if cv > 30:
            recommendations.append(
                f"Revenue varies significantly month-to-month ({cv:.0f}% variation) - "
                f"use Demand Forecasting and Seasonality Analysis to smooth capacity planning"
            )
        
        # Customer retention
        recommendations.append(
            "Implement formal follow-up program for maintenance - "
            "converting one-time customers to recurring maintenance contracts stabilizes revenue"
        )
        
        # Payment methods
        if self.transactions_df['payment_method'].value_counts().get('Financing', 0) < 5:
            recommendations.append(
                "Consider expanding financing options for high-ticket installations - "
                "can increase conversion on large jobs"
            )
        
        recommendations.append(
            "**Next step**: Review Customer Segmentation analysis to understand which customer groups "
            "drive different service categories and tailor strategies accordingly"
        )
        
        return recommendations
    
    # Backward compatibility
    @property
    def recommendations(self) -> list:
        return self.get_recommendations()
    
    @property
    def business_impact(self) -> str:
        return "Understanding the current state provides the foundation for all optimization efforts - knowing what drives revenue today informs where to focus improvement efforts tomorrow"
