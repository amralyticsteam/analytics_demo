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
        fig.update_yaxes(title_text="Profit Margin (%)", row=1, col=1)
        
        fig.update_xaxes(title_text="Service Category", row=1, col=2, tickangle=-45)
        fig.update_yaxes(title_text="Average Cost ($)", row=1, col=2)
        
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
        fig.update_yaxes(title_text="Service", row=2, col=2)
        
        return fig
    
    def get_insights(self) -> list:
        """Generate data-driven insights from pricing analysis."""
        if self.menu_df is None:
            self.load_data()
        
        insights = []
        
        # Menu engineering quadrants
        stars = self.menu_df[self.menu_df['quadrant'] == 'Star']
        plowhorses = self.menu_df[self.menu_df['quadrant'] == 'Plowhorse']
        dogs = self.menu_df[self.menu_df['quadrant'] == 'Dog']
        
        if len(stars) > 0:
            insights.append(
                f"**Stars (high profit + high popularity)**: {len(stars)} services including "
                f"{', '.join(stars.nlargest(3, 'popularity_pct')['service_name'].tolist())} - "
                f"promote these heavily!"
            )
        
        if len(plowhorses) > 0:
            top_plowhorse = plowhorses.nlargest(1, 'popularity_count').iloc[0]
            insights.append(
                f"**Critical issue - Plowhorses**: '{top_plowhorse['service_name']}' is your most popular service "
                f"with {top_plowhorse['popularity_pct']:.1f}% of jobs, but only {top_plowhorse['margin_pct']:.1f}% margin. "
                f"**Must raise price immediately**"
            )
        
        # Zero/negative margin services
        unprofitable = self.menu_df[self.menu_df['margin_pct'] <= 0]
        if len(unprofitable) > 0:
            total_unprofitable_jobs = unprofitable['popularity_count'].sum()
            total_jobs = self.menu_df['popularity_count'].sum()
            pct_unprofitable = (total_unprofitable_jobs / total_jobs) * 100
            
            insights.append(
                f"**{len(unprofitable)} services have 0% or negative margins**, representing "
                f"{pct_unprofitable:.1f}% of all jobs. These are losing money with every service call!"
            )
        
        # Labor efficiency
        most_efficient = self.labor_analysis.iloc[0]
        least_efficient = self.labor_analysis.iloc[-1]
        
        insights.append(
            f"**Labor efficiency varies widely**: {most_efficient['category']} generates "
            f"${most_efficient['revenue_per_labor_hour']:.0f}/hour vs {least_efficient['category']} "
            f"at ${least_efficient['revenue_per_labor_hour']:.0f}/hour"
        )
        
        # COGS insights
        highest_cogs = self.cogs_analysis.iloc[0]
        insights.append(
            f"{highest_cogs['category']} services have highest average COGS at "
            f"${highest_cogs['avg_total_cogs']:.0f}, with {highest_cogs['avg_margin']:.1f}% average margin"
        )
        
        # Top 10 popularity insight
        top_10 = self.menu_df.nlargest(10, 'popularity_count')
        top_10_low_margin = top_10[top_10['margin_pct'] < 5]
        if len(top_10_low_margin) > 0:
            insights.append(
                f"**Top 10 popular services**: {len(top_10_low_margin)} have margins below 5% - "
                f"color-coded visualization shows red/orange for unprofitable popular services"
            )
        
        # Connection to other analyses
        insights.append(
            "**Connection to Demand Forecasting**: Use pricing insights to optimize capacity - "
            f"promote high-margin services during slow periods to improve overall profitability"
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
        
        # Plowhorse pricing
        plowhorses = self.menu_df[self.menu_df['quadrant'] == 'Plowhorse'].nlargest(3, 'popularity_count')
        if len(plowhorses) > 0:
            recommendations.append(
                f"**URGENT - Raise prices on popular services**: "
                f"{', '.join(plowhorses['service_name'].tolist())} need 10-15% price increases. "
                f"These are your most common jobs - small price increase = big revenue impact"
            )
        
        # Stars promotion
        stars = self.menu_df[self.menu_df['quadrant'] == 'Star']
        if len(stars) > 0:
            recommendations.append(
                f"**Promote your Stars**: Feature {', '.join(stars.nlargest(2, 'margin_pct')['service_name'].tolist())} "
                f"in marketing - high profit + high demand = perfect combination"
            )
        
        # Puzzles conversion
        puzzles = self.menu_df[self.menu_df['quadrant'] == 'Puzzle']
        if len(puzzles) > 0:
            recommendations.append(
                f"**Convert Puzzles to Stars**: {len(puzzles)} profitable but uncommon services. "
                f"Create bundled packages or special promotions to increase their popularity"
            )
        
        # Dogs elimination
        dogs = self.menu_df[self.menu_df['quadrant'] == 'Dog']
        if len(dogs) > 0:
            recommendations.append(
                f"**Eliminate or reprice Dogs**: {len(dogs)} services with low profit and low popularity. "
                f"Either raise prices 20%+ or stop offering them"
            )
        
        # Labor optimization
        recommendations.append(
            "Schedule high-revenue-per-hour services during peak technician availability. "
            "Use slow periods for training or maintenance tasks"
        )
        
        recommendations.append(
            "**Next step**: Cross-reference pricing changes with Customer Segmentation to ensure "
            "price-sensitive segments aren't disproportionately affected"
        )
        
        return recommendations
    
    # Backward compatibility
    @property
    def recommendations(self) -> list:
        return self.get_recommendations()
    
    @property
    def business_impact(self) -> str:
        return "Optimizing pricing on just the top 10 services could increase gross profit by $30K-50K annually. Even a 10% price increase on popular zero-margin services would transform profitability without losing customers"
