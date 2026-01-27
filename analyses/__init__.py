"""Analytics modules for the demo application."""

# Core modules
from .base_analysis import BaseAnalysis
from .business_overview import BusinessOverview
from .customer_segmentation import CustomerSegmentation
from .sentiment_analysis import SentimentAnalysis
from .topic_extraction import TopicExtraction
from .pricing_analysis import PricingAnalysis
from .demand_forecasting import DemandForecasting
from .seasonality_timeseries import SeasonalityTimeSeries
from .basket_analysis import BasketAnalysis
from .marketing_impact import MarketingImpact
from .churn_modeling import ChurnModeling

__all__ = [
    'BaseAnalysis',
    'BusinessOverview',
    'CustomerSegmentation',
    'SentimentAnalysis',
    'TopicExtraction',
    'PricingAnalysis',
    'DemandForecasting',
    'SeasonalityTimeSeries',
    'BasketAnalysis',
    'MarketingImpact',
    'ChurnModeling',
]

