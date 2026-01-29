"""Pricing & Menu Analysis Module - Amralytics Methodology"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .base_analysis import BaseAnalysis


class PricingAnalysis(BaseAnalysis):
    """Menu engineering, pricing optimization, and profitability analysis"""
    
    def __init__(self):
        self.menu_df = None
        self.menu_quadrants = None
        self.cogs_analysis = None
        self.labor_analysis = None
    
    @property
    def icon(self) -> str:
        return '💰'
    
    @property
    def color(self) -> str:
        return '#008f8c'
    
    @property
    def rons_challenge(self) -> str:
        return """Ron hasn't reviewed his pricing in over 2 years. He's been so focused on finding new 
        customers that he hasn't checked if his most popular services are even profitable.
        
        **Are his prices covering costs?** More importantly: which services are profitable, which are 
        losing money, and which should he promote more heavily? Ron needs a pricing strategy, not just 
        price tags."""
    
    # Backward compatibility
    @property
    def business_question(self) -> str:
        return self.rons_challenge
    
    @property
    def data_collected(self) -> list:
        return [
            'Service pricing (50 HVAC services) - **QuickBooks**',
            'Parts costs by service - **QuickBooks**',
            'Labor hours and rates - **ServiceTitan**',
            'Service popularity/frequency - **ServiceTitan**',
            'Gross margins by service - **QuickBooks**'
        ]
    
    
    # Backward compatibility
    @property
    def data_inputs(self) -> list:
        return self.data_collected
    
    @property
    def methodology(self) -> str:
        return 'Menu Engineering Matrix (popularity vs profitability quadrant analysis), COGS breakdown by service category, labor ratio analysis, price elasticity assessment, competitive positioning'
    
    # Backward compatibility
    @property
    def technical_output(self) -> str:
        return self.methodology
    
    @property
    def data_file(self) -> str:
        return 'pricing_menu.csv'
    
    def load_data(self, filepath: str = None):
        """Load and process pricing/menu data."""
        if filepath is None:
            filepath = f'data/{self.data_file}'
        
        self.menu_df = pd.read_csv(filepath)
        
        # Calculate additional metrics
        self.menu_df['total_labor_cost'] = (
            self.menu_df['cogs_labor_hours'] * self.menu_df['labor_rate']
        )
        self.menu_df['parts_pct_of_cogs'] = (
            self.menu_df['cogs_parts'] / self.menu_df['total_cogs'] * 100
        )
        self.menu_df['labor_pct_of_cogs'] = (
            self.menu_df['total_labor_cost'] / self.menu_df['total_cogs'] * 100
        )
        
        # Classify into menu engineering quadrants
        self._classify_menu_items()
        
        # Analyze COGS
        self._analyze_cogs()
        
        # Analyze labor
        self._analyze_labor()
        
        return self.menu_df
    
    def _classify_menu_items(self):
        """Classify services into menu engineering quadrants (Stars, Plowhorses, Puzzles, Dogs)."""
        # Define thresholds
        median_popularity = self.menu_df['popularity_pct'].median()
        median_margin = self.menu_df['margin_pct'].median()
        
        def classify_item(row):
            high_popularity = row['popularity_pct'] > median_popularity
            high_margin = row['margin_pct'] > median_margin
            
            if high_margin and high_popularity:
                return 'Star'
            elif not high_margin and high_popularity:
                return 'Plowhorse'
            elif high_margin and not high_popularity:
                return 'Puzzle'
            else:
                return 'Dog'
        
        self.menu_df['quadrant'] = self.menu_df.apply(classify_item, axis=1)
        
        # Store summary by quadrant
        self.menu_quadrants = self.menu_df.groupby('quadrant').agg({
            'service_name': 'count',
            'price': 'mean',
            'margin_pct': 'mean',
            'popularity_pct': 'mean'
        }).reset_index()
        self.menu_quadrants.columns = ['quadrant', 'count', 'avg_price', 'avg_margin', 'avg_popularity']
    
    def _analyze_cogs(self):
        """Analyze cost of goods sold by category."""
        self.cogs_analysis = self.menu_df.groupby('category').agg({
            'cogs_parts': 'mean',
            'total_labor_cost': 'mean',
            'total_cogs': 'mean',
            'margin_pct': 'mean',
            'service_name': 'count'
        }).reset_index()
        self.cogs_analysis.columns = [
            'category', 'avg_parts_cost', 'avg_labor_cost', 
            'avg_total_cogs', 'avg_margin', 'service_count'
        ]
        self.cogs_analysis = self.cogs_analysis.sort_values('avg_total_cogs', ascending=False)
    
    def _analyze_labor(self):
        """Analyze labor ratios and efficiency."""
        self.labor_analysis = self.menu_df.groupby('category').agg({
            'cogs_labor_hours': 'mean',
            'labor_pct_of_cogs': 'mean',
            'service_name': 'count',
            'price': 'mean'
        }).reset_index()
        self.labor_analysis.columns = [
            'category', 'avg_labor_hours', 'labor_pct_of_cogs', 
            'service_count', 'avg_price'
        ]
        self.labor_analysis['revenue_per_labor_hour'] = (
            self.labor_analysis['avg_price'] / self.labor_analysis['avg_labor_hours']
        )
        self.labor_analysis = self.labor_analysis.sort_values('revenue_per_labor_hour', ascending=False)
    
    def create_visualization(self):
        """Create 4-panel pricing analysis dashboard."""
        if self.menu_df is None:
            self.load_data()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Menu Engineering Matrix: Popularity vs Profitability',
                'COGS Breakdown by Service Category',
                'Labor Efficiency: Revenue per Labor Hour',
                'Top 10 Services by Popularity (colored by margin)'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}]
            ],
            vertical_spacing=0.25,
            horizontal_spacing=0.18,  # Increased from 0.15
            column_widths=[0.45, 0.45],  # Equal widths but creates more space
            row_heights=[0.48, 0.48]
        )
        
        # 1. Menu Engineering Matrix - Scatter plot
        category_colors = {
            'Cooling': '#008f8c',
            'Heating': '#ff6b6b',
            'Maintenance': '#23606e',
            'Emergency': '#ffa07a',
            'Installation': '#00b894'
        }
        
        quadrant_shapes = {
            'Star': 'circle',
            'Plowhorse': 'square',
            'Puzzle': 'diamond',
            'Dog': 'x'
        }
        
        # Track which categories we've added to legend
        legend_added = set()
        
        for category in self.menu_df['category'].unique():
            category_data = self.menu_df[self.menu_df['category'] == category]
            
            for quadrant in category_data['quadrant'].unique():
                quad_data = category_data[category_data['quadrant'] == quadrant]
                
                # Only show legend once per category (not for each quadrant)
                show_in_legend = category not in legend_added
                if show_in_legend:
                    legend_added.add(category)
                
                fig.add_trace(
                    go.Scatter(
                        x=quad_data['popularity_pct'],
                        y=quad_data['margin_pct'],
                        mode='markers',
                        name=f"{category}",
                        legendgroup=category,  # Group by category
                        marker=dict(
                            size=quad_data['price'] / 30,
                            color=category_colors.get(category, '#023535'),
                            symbol=quadrant_shapes.get(quadrant, 'circle'),
                            line=dict(width=2, color='white')
                        ),
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Category: " + category + "<br>"
                            "Quadrant: " + quadrant + "<br>"
                            "Popularity: %{x:.1f}%<br>"
                            "Margin: %{y:.1f}%<br>"
                            "Price: $%{customdata[1]:.0f}<extra></extra>"
                        ),
                        customdata=quad_data[['service_name', 'price']].values,
                        showlegend=show_in_legend
                    ),
                    row=1, col=1
                )
        
        # Add quadrant divider lines
        median_pop = self.menu_df['popularity_pct'].median()
        median_margin = self.menu_df['margin_pct'].median()
        
        fig.add_vline(x=median_pop, line_dash="dash", line_color="#023535", 
                     opacity=0.5, row=1, col=1)
        fig.add_hline(y=median_margin, line_dash="dash", line_color="#023535", 
                     opacity=0.5, row=1, col=1)
        
        # 2. COGS Breakdown - Stacked bar chart
        fig.add_trace(
            go.Bar(
                x=self.cogs_analysis['category'],
                y=self.cogs_analysis['avg_parts_cost'],
                name='Parts Cost',
                marker_color='#ff6b6b',
                text=[f"${v:.0f}" for v in self.cogs_analysis['avg_parts_cost']],
                textposition='inside',  # Changed to inside
                textfont=dict(size=10)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=self.cogs_analysis['category'],
                y=self.cogs_analysis['avg_labor_cost'],
                name='Labor Cost',
                marker_color='#008f8c',
                text=[f"${v:.0f}" for v in self.cogs_analysis['avg_labor_cost']],
                textposition='inside',  # Changed to inside
                textfont=dict(size=10)
            ),
            row=1, col=2
        )
        
        # 3. Labor Efficiency - Revenue per labor hour
        fig.add_trace(
            go.Bar(
                x=self.labor_analysis['category'],
                y=self.labor_analysis['revenue_per_labor_hour'],
                marker_color='#23606e',
                text=[f"${v:.0f}" for v in self.labor_analysis['revenue_per_labor_hour']],
                textposition='outside',
                textfont=dict(size=10),
                showlegend=False,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Revenue/Hour: $%{y:.0f}<br>"
                    "Avg Labor Hours: %{customdata[0]:.1f}<br>"
                    "Avg Price: $%{customdata[1]:.0f}<extra></extra>"
                ),
                customdata=self.labor_analysis[['avg_labor_hours', 'avg_price']],
                cliponaxis=False  # Allow text to extend beyond plot area
            ),
            row=2, col=1
        )
        
        # 4. Top 10 Services - Colored by margin
        top_10 = self.menu_df.nlargest(10, 'popularity_count').sort_values('popularity_count')  # Sort ascending for better display
        
        # Create color scale based on margin
        colors = []
        for margin in top_10['margin_pct']:
            if margin < -10:
                colors.append('#d63031')
            elif margin < 0:
                colors.append('#ff6b6b')
            elif margin < 5:
                colors.append('#ffa07a')
            elif margin < 10:
                colors.append('#98d8c8')
            else:
                colors.append('#00b894')
        
        fig.add_trace(
            go.Bar(
                y=top_10['service_name'],
                x=top_10['popularity_count'],
                orientation='h',
                marker_color=colors,
                text=[f"{m:.1f}%" for m in top_10['margin_pct']],
                textposition='outside',
                textfont=dict(size=9),
                showlegend=False,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Count: %{x} jobs<br>"
                    "Margin: %{customdata[0]:.1f}%<br>"
                    "Price: $%{customdata[1]:.0f}<extra></extra>"
                ),
                customdata=top_10[['margin_pct', 'price']],
                cliponaxis=False
            ),
            row=2, col=2
        )
        
        # Update layout with increased height and better margins
        fig.update_layout(
            height=950,  # Increased from 850
            showlegend=True,
            title_text="Pricing Analysis: Menu Engineering & Profitability",
            title_x=0.5,
            title_font=dict(size=18, color='#023535'),
            barmode='stack',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=1.02,  # Just outside the right edge
                bgcolor='rgba(255,252,242,0.9)',
                bordercolor='#023535',
                borderwidth=1,
                font=dict(size=11)
            ),
            margin=dict(l=80, r=180, t=100, b=80)  # Increased right margin for legend
        )
        
        # Update axes with better formatting
        fig.update_xaxes(title_text="Popularity (% of Jobs)", row=1, col=1)
        fig.update_yaxes(title_text="Profit Margin (%)", row=1, col=1, rangemode="tozero")
        
        fig.update_xaxes(title_text="Service Category", row=1, col=2, tickangle=-45)
        fig.update_yaxes(title_text="Average Cost ($)", row=1, col=2, rangemode="tozero")
        
        fig.update_xaxes(title_text="Service Category", row=2, col=1, tickangle=-45)
        fig.update_yaxes(
            title_text="Revenue per Labor Hour ($)", 
            row=2, col=1,
            range=[0, self.labor_analysis['revenue_per_labor_hour'].max() * 1.15]  # Add 15% padding for text
        )
        
        fig.update_xaxes(
            title_text="Number of Jobs", 
            row=2, col=2,
            range=[0, top_10['popularity_count'].max() * 1.2]  # Add 20% padding for text
        )
        fig.update_yaxes(title_text="Service", row=2, col=2, rangemode="tozero")
        
        return fig
    
    def get_insights(self) -> list:
        """Generate data-driven insights from pricing analysis."""
        if self.menu_df is None:
            self.load_data()
        
        insights = []
        
        # Critical issue - most popular services losing money
        unprofitable = self.menu_df[self.menu_df['margin_pct'] < 0]
        if len(unprofitable) > 0:
            top_unprofitable = unprofitable.nlargest(3, 'popularity_count')
            total_unprofitable_jobs = unprofitable['popularity_count'].sum()
            
            insights.append(
                f"Critical problem: Most popular services are LOSING money - "
                f"{top_unprofitable.iloc[0]['service_name']} ({int(top_unprofitable.iloc[0]['popularity_count'])} jobs/year, "
                f"{top_unprofitable.iloc[0]['margin_pct']:.1f}% margin), "
                f"{top_unprofitable.iloc[1]['service_name']} ({int(top_unprofitable.iloc[1]['popularity_count'])} jobs/year, "
                f"{top_unprofitable.iloc[1]['margin_pct']:.1f}% margin), and "
                f"{top_unprofitable.iloc[2]['service_name']} ({int(top_unprofitable.iloc[2]['popularity_count'])} jobs/year, "
                f"{top_unprofitable.iloc[2]['margin_pct']:.1f}% margin) are top sellers with negative margins"
            )
        
        # Zero/negative margin services count
        if len(unprofitable) > 0:
            insights.append(
                f"{len(unprofitable)} services with negative margins representing "
                f"{total_unprofitable_jobs:.0f} annual jobs - every service call loses money"
            )
        
        # High-margin services underutilized
        high_margin = self.menu_df[self.menu_df['margin_pct'] > 20].nsmallest(3, 'popularity_count')
        if len(high_margin) > 0:
            insights.append(
                f"High-margin services are underutilized - "
                f"{high_margin.iloc[0]['service_name']} ({high_margin.iloc[0]['margin_pct']:.1f}% margin), "
                f"{high_margin.iloc[1]['service_name']} ({high_margin.iloc[1]['margin_pct']:.1f}% margin), and "
                f"{high_margin.iloc[2]['service_name']} ({high_margin.iloc[2]['margin_pct']:.1f}% margin) "
                f"are profitable but low volume"
            )
        
        # Major installations razor-thin margins
        installations = self.menu_df[self.menu_df['category'] == 'Installation']
        low_margin_installs = installations[installations['margin_pct'] < 10]
        if len(low_margin_installs) > 0:
            insights.append(
                f"Major installations have razor-thin margins - {len(low_margin_installs)} installation services "
                f"are at 4-10% margins, leaving no room for overhead (sales costs, permits, callbacks)"
            )
        
        # Emergency services priced correctly
        emergency = self.menu_df[self.menu_df['category'] == 'Emergency']
        if len(emergency) > 0:
            avg_emergency_margin = emergency['margin_pct'].mean()
            insights.append(
                f"Emergency services are priced correctly - Average {avg_emergency_margin:.1f}% margin "
                f"shows Ron understands premium pricing for urgent work"
            )
        
        # Labor rate analysis
        insights.append(
            f"Labor rate may be too high at $85/hour - combined with parts costs, many services "
            f"can't cover fully-loaded labor costs (benefits, insurance, vehicle costs add 30-40% to base wage)"
        )
        
        # Top 10 popularity insight - color-coded visualization
        top_10 = self.menu_df.nlargest(10, 'popularity_count')
        top_10_low_margin = top_10[top_10['margin_pct'] < 10]
        if len(top_10_low_margin) > 0:
            insights.append(
                f"Top 10 popular services: {len(top_10_low_margin)} have margins below 10% - "
                f"color-coded visualization shows red/orange for problem areas requiring immediate attention"
            )
        
        # Connection to other analyses
        insights.append(
            "Connection to Customer Segmentation: Different customer segments have different price sensitivity - "
            f"use segmentation data to target price increases strategically (VIP customers can absorb increases better)"
        )
        
        return insights
    
    # Backward compatibility
    @property
    def insights(self) -> list:
        return self.get_insights()
    
    def get_recommendations(self) -> list:
        """Generate actionable recommendations."""
        if self.menu_df is None:
            self.load_data()
        
        recommendations = []
        
        # URGENT: Fix negative margin services
        unprofitable = self.menu_df[self.menu_df['margin_pct'] < 0].nlargest(3, 'popularity_count')
        if len(unprofitable) > 0:
            recommendations.append(
                f"URGENT: Raise prices on negative-margin services immediately - "
                f"Increase {unprofitable.iloc[0]['service_name']} to ${int(unprofitable.iloc[0]['price'] * 1.19)} "
                f"(+${int(unprofitable.iloc[0]['price'] * 0.19)}, gets to ~18% margin), "
                f"{unprofitable.iloc[1]['service_name']} to ${int(unprofitable.iloc[1]['price'] * 1.18)} "
                f"(+${int(unprofitable.iloc[1]['price'] * 0.18)}, gets to ~16% margin), "
                f"{unprofitable.iloc[2]['service_name']} to ${int(unprofitable.iloc[2]['price'] * 1.18)} "
                f"(+${int(unprofitable.iloc[2]['price'] * 0.18)}, gets to ~20% margin). "
                f"These three changes alone affect {int(unprofitable['popularity_count'].sum())} jobs/year"
            )
        
        # Service bundles
        recommendations.append(
            "Create service bundles to improve margins - Pair low-margin installations with high-margin "
            "add-ons: 'New Thermostat + Calibration' bundle ($315 vs $284 separate, better margin), "
            "'Humidifier Install + Annual Check' package (lock in recurring revenue)"
        )
        
        # Promote high-margin services
        high_margin = self.menu_df[self.menu_df['margin_pct'] > 20].nlargest(3, 'margin_pct')
        if len(high_margin) > 0:
            recommendations.append(
                f"Promote high-margin maintenance services - "
                f"{high_margin.iloc[0]['service_name']} ({high_margin.iloc[0]['margin_pct']:.1f}% margin), "
                f"{high_margin.iloc[1]['service_name']} ({high_margin.iloc[1]['margin_pct']:.1f}% margin), and "
                f"{high_margin.iloc[2]['service_name']} ({high_margin.iloc[2]['margin_pct']:.1f}% margin) "
                f"have 30-45% margins. Market these aggressively through service contracts and reminders"
            )
        
        # Labor rate structure
        recommendations.append(
            "Review labor rate structure - $85/hour may be too high for simple maintenance work. "
            "Consider tiered rates: $65/hr for routine maintenance, $85/hr for repairs, $100/hr for installations. "
            "This would make many maintenance services profitable while remaining competitive"
        )
        
        # Parts negotiation
        recommendations.append(
            "Renegotiate parts costs for major installations - Big-ticket items (AC, Furnace, Heat Pump) "
            "have 4-8% margins. Work with suppliers to reduce parts costs by 10-15% or increase prices proportionally. "
            "Alternative: add installation fee separate from equipment cost to improve transparency and margins"
        )
        
        # Annual price increases
        recommendations.append(
            "Implement annual price review process - Set calendar reminder for January to review all prices. "
            "Implement 3-5% annual increases to keep up with inflation and cost increases. "
            "Communicate changes to customers 30 days in advance with explanation (rising costs, better service)"
        )
        
        # Connection to other analyses
        recommendations.append(
            "Next step: Cross-reference pricing changes with Customer Segmentation to ensure "
            "price-sensitive segments aren't disproportionately affected. Use Churn Modeling to identify "
            "customers at risk of leaving before implementing increases"
        )
        
        return recommendations
    
    # Backward compatibility
    @property
    def recommendations(self) -> list:
        return self.get_recommendations()
    
    @property
    def business_impact(self) -> str:
        return "Fixing pricing on the 7 negative-margin services would stop the bleeding immediately. A 15-20% price increase on these services (346 annual jobs) would add $18K-25K to gross profit annually - the difference between breaking even and sustainable profitability"
