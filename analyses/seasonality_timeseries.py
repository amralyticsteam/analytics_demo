"""Seasonality & Time Series Analysis Module - Amralytics Methodology"""

import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .base_analysis import BaseAnalysis


class SeasonalityTimeSeries(BaseAnalysis):
    """Seasonality analysis with service type breakdown"""
    
    def __init__(self):
        self.timeseries_df = None
        self.decomposition = None
        self.service_seasonality = None
    
    @property
    def icon(self) -> str:
        return '📅'
    
    @property
    def color(self) -> str:
        return '#23606e'
    
    @property
    def rons_challenge(self) -> str:
        return """Ron knows his business has busy and slow periods, but he doesn't have a clear picture 
        of **normal seasonal patterns vs concerning trends**.
        
        Is that drop in October normal? Should July AC revenue be higher? **Understanding seasonality 
        by service type** helps Ron plan staffing, inventory, and marketing spend throughout the year."""
    
    # Backward compatibility
    @property
    def business_question(self) -> str:
        return self.rons_challenge
    
        @property
    def data_collected(self) -> list:
        return [
            '**Source**: QuickBooks Revenue Reports',
            '**Dataset**: seasonality_timeseries.csv',
            '**Records**: 32 months (Jan 2022 - Aug 2024)',
            '**Contains**: Monthly revenue by service type (Installation, Maintenance, Emergency, Cooling, Heating)'
        ]
    
    
    # Backward compatibility
    @property
    def data_inputs(self) -> list:
        return self.data_collected
    
    @property
    def methodology(self) -> str:
        return 'Time series decomposition (trend + seasonal + residual), service-specific seasonal patterns, year-over-year growth analysis, seasonal indices by category'
    
    # Backward compatibility
    @property
    def technical_output(self) -> str:
        return self.methodology
    
    @property
    def data_file(self) -> str:
        return 'seasonality_timeseries.csv'
    
    def load_data(self, filepath: str = None):
        """Load and process time series data."""
        if filepath is None:
            filepath = f'data/{self.data_file}'
        
        self.timeseries_df = pd.read_csv(filepath)
        
        # Create proper date column - month column already contains YYYY-MM format
        self.timeseries_df['date'] = pd.to_datetime(self.timeseries_df['month'])
        
        # Sort by date
        self.timeseries_df = self.timeseries_df.sort_values('date')
        
        # Add month name and year for display
        self.timeseries_df['month_name'] = self.timeseries_df['date'].dt.strftime('%b')
        self.timeseries_df['year'] = self.timeseries_df['date'].dt.year
        self.timeseries_df['year_str'] = self.timeseries_df['date'].dt.year.astype(str)
        
        # Calculate service-specific seasonality
        self._calculate_service_seasonality()
        
        return self.timeseries_df
    
    def _calculate_service_seasonality(self):
        """Calculate seasonal patterns for each service type."""
        # Get month number
        self.timeseries_df['month_num'] = self.timeseries_df['date'].dt.month
        
        # Calculate average by month across all years for each service
        service_columns = [
            'cooling', 'heating', 
            'maintenance', 'emergency', 'installation'
        ]
        
        monthly_avgs = []
        for month in range(1, 13):
            month_data = self.timeseries_df[self.timeseries_df['month_num'] == month]
            
            if len(month_data) == 0:
                continue
                
            row = {'month': month, 'month_name': month_data.iloc[0]['month_name']}
            for col in service_columns:
                if col in month_data.columns:
                    row[col] = month_data[col].mean()
                else:
                    row[col] = 0
            
            monthly_avgs.append(row)
        
        self.service_seasonality = pd.DataFrame(monthly_avgs)
    
    def create_visualization(self):
        """Create 4-panel seasonality dashboard."""
        if self.timeseries_df is None:
            self.load_data()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Revenue by Service Type Over Time',
                'Monthly Seasonal Patterns by Service',
                'Year-over-Year Revenue Comparison',
                'Service Mix: Revenue Distribution'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}]
            ],
            vertical_spacing=0.22,
            horizontal_spacing=0.15
        )
        
        # Service colors
        service_colors = {
            'AC Repair': '#008f8c',
            'Heating Repair': '#ff6b6b',
            'Maintenance': '#23606e',
            'Emergency': '#ffa07a',
            'Installation': '#00b894'
        }
        
        # 1. Revenue Over Time - Stacked area or lines
        services = [
            ('cooling', 'Cooling'),
            ('heating', 'Heating'),
            ('maintenance', 'Maintenance'),
            ('emergency', 'Emergency'),
            ('installation', 'Installation')
        ]
        
        for col, name in services:
            fig.add_trace(
                go.Scatter(
                    x=self.timeseries_df['date'],
                    y=self.timeseries_df[col],
                    name=name,
                    mode='lines',
                    line=dict(width=2, color=service_colors[name]),
                    stackgroup='one',  # Creates stacked area chart
                    hovertemplate='%{x|%b %Y}<br>' + name + ': $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Monthly Seasonal Patterns
        if self.service_seasonality is not None:
            for col, name in services:
                fig.add_trace(
                    go.Scatter(
                        x=self.service_seasonality['month_name'],
                        y=self.service_seasonality[col],
                        name=name,
                        mode='lines+markers',
                        line=dict(width=3, color=service_colors[name]),
                        marker=dict(size=8),
                        hovertemplate=name + '<br>%{x}: $%{y:,.0f}<extra></extra>',
                        showlegend=False  # Already shown in first chart
                    ),
                    row=1, col=2
                )
        
        # 3. Year-over-Year Comparison - Grouped bars by year
        years = self.timeseries_df['year'].unique()
        
        # Get last 12 months for each year (or all available)
        for year in years:
            year_data = self.timeseries_df[self.timeseries_df['year'] == year]
            
            fig.add_trace(
                go.Bar(
                    x=year_data['month_name'],
                    y=year_data['total_revenue'],
                    name=str(year),
                    marker_color='#008f8c' if year == years[-1] else '#98d8c8',
                    text=[f"${v/1000:.0f}K" for v in year_data['total_revenue']],
                    textposition='outside',
                    textfont=dict(size=9),
                    hovertemplate='%{x} ' + str(year) + '<br>Revenue: $%{y:,.0f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Service Mix - Average revenue distribution
        avg_by_service = []
        for col, name in services:
            avg_by_service.append({
                'service': name,
                'avg_revenue': self.timeseries_df[col].mean()
            })
        
        mix_df = pd.DataFrame(avg_by_service).sort_values('avg_revenue', ascending=True)
        
        colors_list = [service_colors[s] for s in mix_df['service']]
        
        fig.add_trace(
            go.Bar(
                y=mix_df['service'],
                x=mix_df['avg_revenue'],
                orientation='h',
                marker_color=colors_list,
                text=[f"${v/1000:.0f}K" for v in mix_df['avg_revenue']],
                textposition='outside',
                showlegend=False,
                hovertemplate='<b>%{y}</b><br>Avg Monthly Revenue: $%{x:,.0f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text="Seasonality Analysis: Service-Specific Patterns",
            title_x=0.5,
            title_font=dict(size=18, color='#023535'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            barmode='group',
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Revenue ($)", row=1, col=1, rangemode="tozero")
        
        fig.update_xaxes(title_text="Month", row=1, col=2)
        fig.update_yaxes(title_text="Avg Revenue ($)", row=1, col=2, rangemode="tozero")
        
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Total Revenue ($)", row=2, col=1, rangemode="tozero")
        
        fig.update_xaxes(title_text="Avg Monthly Revenue ($)", row=2, col=2)
        fig.update_yaxes(title_text="Service Type", row=2, col=2, rangemode="tozero")
        
        return fig
    
    def get_insights(self) -> list:
        """Generate data-driven insights from seasonality analysis."""
        if self.timeseries_df is None:
            self.load_data()
        
        insights = []
        
        # Peak months by service
        ac_peak_month = self.service_seasonality.loc[
            self.service_seasonality['cooling'].idxmax(), 'month_name'
        ]
        heating_peak_month = self.service_seasonality.loc[
            self.service_seasonality['heating'].idxmax(), 'month_name'
        ]
        
        ac_peak_value = self.service_seasonality['cooling'].max()
        heating_peak_value = self.service_seasonality['heating'].max()
        
        insights.append(
            f"**Clear seasonal patterns**: Cooling peaks in {ac_peak_month} (${ac_peak_value:,.0f}), "
            f"Heating peaks in {heating_peak_month} (${heating_peak_value:,.0f})"
        )
        
        # Maintenance steadiness
        maintenance_std = self.service_seasonality['maintenance'].std()
        maintenance_mean = self.service_seasonality['maintenance'].mean()
        maintenance_cv = (maintenance_std / maintenance_mean) * 100
        
        if maintenance_cv < 15:
            insights.append(
                f"Maintenance revenue is steady year-round (variation: {maintenance_cv:.1f}%) - "
                f"provides reliable baseline income"
            )
        
        # Year-over-year trend
        recent_year = self.timeseries_df['year'].max()
        prior_year = recent_year - 1
        
        recent_total = self.timeseries_df[
            self.timeseries_df['year'] == recent_year
        ]['total_revenue'].sum()
        prior_total = self.timeseries_df[
            self.timeseries_df['year'] == prior_year
        ]['total_revenue'].sum()
        
        yoy_change = ((recent_total - prior_total) / prior_total) * 100
        
        insights.append(
            f"Year-over-year trend: {recent_year} total revenue is "
            f"{'up' if yoy_change > 0 else 'down'} {abs(yoy_change):.1f}% vs {prior_year} "
            f"(${recent_total:,.0f} vs ${prior_total:,.0f})"
        )
        
        # Service mix
        avg_service_revenue = {
            'Cooling': self.timeseries_df['cooling'].mean(),
            'Heating': self.timeseries_df['heating'].mean(),
            'Maintenance': self.timeseries_df['maintenance'].mean(),
            'Emergency': self.timeseries_df['emergency'].mean(),
            'Installation': self.timeseries_df['installation'].mean()
        }
        
        top_service = max(avg_service_revenue, key=avg_service_revenue.get)
        insights.append(
            f"**Largest revenue driver**: {top_service} averages "
            f"${avg_service_revenue[top_service]:,.0f}/month"
        )
        
        # Emergency revenue correlation with extremes
        summer_months = self.timeseries_df[self.timeseries_df['date'].dt.month.isin([6, 7, 8])]
        winter_months = self.timeseries_df[self.timeseries_df['date'].dt.month.isin([12, 1, 2])]
        
        summer_emergency = summer_months['emergency'].mean()
        winter_emergency = winter_months['emergency'].mean()
        
        if max(summer_emergency, winter_emergency) > 1.3 * min(summer_emergency, winter_emergency):
            peak_season = "summer" if summer_emergency > winter_emergency else "winter"
            insights.append(
                f"Emergency services spike in {peak_season} "
                f"(${max(summer_emergency, winter_emergency):,.0f} vs "
                f"${min(summer_emergency, winter_emergency):,.0f}) - "
                f"ensure adequate staffing"
            )
        
        # Connection to other analyses
        insights.append(
            "**Connection to Demand Forecasting**: Seasonal patterns inform capacity planning - "
            "use demand forecasts to fine-tune staffing within these seasonal windows"
        )
        
        return insights
    
    # Backward compatibility
    @property
    def insights(self) -> list:
        return self.get_insights()
    
    def get_recommendations(self) -> list:
        """Generate actionable recommendations."""
        if self.timeseries_df is None:
            self.load_data()
        
        recommendations = []
        
        # Peak season prep
        ac_peak_month = self.service_seasonality.loc[
            self.service_seasonality['cooling'].idxmax(), 'month_name'
        ]
        heating_peak_month = self.service_seasonality.loc[
            self.service_seasonality['heating'].idxmax(), 'month_name'
        ]
        
        recommendations.append(
            f"**Hire seasonal help before peak months**: Bring on extra technicians in May "
            f"(before {ac_peak_month} cooling peak) and November (before {heating_peak_month} heating peak)"
        )
        
        # Inventory management
        recommendations.append(
            "Stock up on AC parts (capacitors, refrigerant) in April-May, "
            "heating parts (igniters, gas valves) in October-November"
        )
        
        # Marketing timing
        recommendations.append(
            "Launch AC tune-up marketing campaigns in March-April (before busy season), "
            "heating tune-up campaigns in September-October"
        )
        
        # Slow season utilization
        recommendations.append(
            "Use slow periods (Spring/Fall) for: training, equipment maintenance, "
            "following up with at-risk customers (see Churn Analysis)"
        )
        
        # Service bundling
        recommendations.append(
            "Create seasonal bundles: 'Summer Prep Package' (AC tune-up + filter), "
            "'Winter Ready Package' (heating tune-up + safety inspection)"
        )
        
        recommendations.append(
            "**Next step**: Use pricing analysis to ensure seasonal services are priced for profitability"
        )
        
        return recommendations
    
    # Backward compatibility
    @property
    def recommendations(self) -> list:
        return self.get_recommendations()
    
    @property
    def business_impact(self) -> str:
        return "Understanding seasonal patterns allows Ron to optimize staffing, inventory, and marketing spend throughout the year, reducing waste during slow periods and maximizing revenue during peaks"
