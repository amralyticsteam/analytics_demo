"""
Template for Creating a New Analysis Module

Copy this file and customize it to add your own analysis type.
File naming: use lowercase with underscores (e.g., price_optimization.py)
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .base_analysis import BaseAnalysis


class TemplateAnalysis(BaseAnalysis):
    """
    [REPLACE] Brief description of what this analysis does.
    
    Example: Optimize pricing strategy based on demand elasticity and competitor analysis.
    """
    
    @property
    def icon(self) -> str:
        """[REPLACE] Return an emoji icon for this analysis."""
        return '🎯'  # Choose an appropriate emoji
    
    @property
    def color(self) -> str:
        """[REPLACE] Return the primary color (hex code) for this analysis."""
        return '#3b82f6'  # Blue - choose your brand color
    
    @property
    def business_question(self) -> str:
        """[REPLACE] What business problem does this solve?"""
        return 'What is the optimal pricing strategy to maximize revenue while maintaining market share?'
    
    @property
    def data_inputs(self) -> list:
        """[REPLACE] List of required data sources."""
        return [
            'Historical pricing data',
            'Sales volume by price point',
            'Competitor pricing',
            'Customer demographics',
            'Market conditions'
        ]
    
    @property
    def technical_output(self) -> str:
        """[REPLACE] Brief description of your analytical methodology."""
        return 'Price elasticity modeling using regression analysis with competitor pricing as covariates'
    
    @property
    def insights(self) -> list:
        """[REPLACE] List 3-5 key insights from your analysis."""
        return [
            'Current pricing is 8% below optimal, leaving $45K monthly revenue on table',
            'Premium segment shows low price sensitivity (elasticity: -0.3)',
            'Competitor price changes have 2-3 week lag in affecting our sales',
            'Weekend pricing can be 12% higher without volume impact'
        ]
    
    @property
    def recommendations(self) -> list:
        """[REPLACE] List 3-5 actionable recommendations."""
        return [
            'Increase base pricing by 8% starting with premium products',
            'Implement dynamic weekend pricing at +12% premium',
            'Create price monitoring system for top 3 competitors',
            'Test promotional pricing during mid-week slow periods'
        ]
    
    @property
    def business_impact(self) -> str:
        """[REPLACE] Expected business impact of implementing recommendations."""
        return 'Projected revenue increase of $540K annually (8% lift) with minimal customer attrition'
    
    @property
    def data_file(self) -> str:
        """[REPLACE] Name of the CSV file in the data/ folder."""
        return 'template_analysis.csv'
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        [CUSTOMIZE IF NEEDED] Load and prepare data for analysis.
        
        Default implementation loads CSV. Override if you need custom preprocessing.
        """
        self.data = pd.read_csv(filepath)
        # Add any data preprocessing here
        # self.data['date'] = pd.to_datetime(self.data['date'])
        return self.data
    
    def create_visualization(self):
        """
        [REPLACE] Create the main visualization for this analysis.
        
        Should return a Plotly figure object (px or go).
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Example 1: Simple bar chart
        fig = px.bar(
            self.data,
            x='category',
            y='value',
            title='Your Analysis Title',
            color='category'
        )
        
        # Example 2: Line chart
        # fig = px.line(
        #     self.data,
        #     x='date',
        #     y='metric',
        #     title='Trend Over Time'
        # )
        
        # Example 3: Scatter plot
        # fig = px.scatter(
        #     self.data,
        #     x='variable_1',
        #     y='variable_2',
        #     size='importance',
        #     color='category'
        # )
        
        # Example 4: Multiple traces with go.Figure
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=self.data['x'], y=self.data['y1'], name='Series 1'))
        # fig.add_trace(go.Scatter(x=self.data['x'], y=self.data['y2'], name='Series 2'))
        # fig.update_layout(title='Your Title')
        
        return fig


# TO ADD THIS ANALYSIS TO THE APP:
# 
# 1. Save this file as analyses/your_analysis_name.py
# 2. Update analyses/__init__.py:
#    from .your_analysis_name import YourAnalysisName
#    Add 'YourAnalysisName' to __all__ list
# 
# 3. Update analytics_showcase_refactored.py:
#    from analyses import (..., YourAnalysisName)
#    Add to ANALYSES dict:
#    'Your Analysis Display Name': YourAnalysisName(),
# 
# 4. Create data/your_data_file.csv with sample data
# 
# 5. Run: streamlit run analytics_showcase_refactored.py
