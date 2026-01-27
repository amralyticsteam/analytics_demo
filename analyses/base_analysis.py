"""Base analysis class that all analysis modules inherit from."""

from abc import ABC, abstractmethod
import pandas as pd


class BaseAnalysis(ABC):
    """Abstract base class for all analysis modules."""
    
    def __init__(self):
        self.data = None
    
    @property
    @abstractmethod
    def icon(self) -> str:
        """Return the emoji/icon for this analysis."""
        pass
    
    @property
    @abstractmethod
    def color(self) -> str:
        """Return the primary color for this analysis (hex code)."""
        pass
    
    @property
    def rons_challenge(self) -> str:
        """Return Ron's business challenge (preferred name)."""
        return self.business_question
    
    @property
    @abstractmethod
    def business_question(self) -> str:
        """Return the business question this analysis answers."""
        pass
    
    @property
    def data_collected(self) -> list:
        """Return list of data collected (preferred name)."""
        return self.data_inputs
    
    @property
    @abstractmethod
    def data_inputs(self) -> list:
        """Return list of data inputs required."""
        pass
    
    @property
    @abstractmethod
    def methodology(self) -> str:
        """Return the methodology/technical approach (preferred name)."""
        pass
    
    @property
    def technical_output(self) -> str:
        """Return the technical output (backward compatibility)."""
        return self.methodology
    
    @property
    def insights(self) -> list:
        """Return list of insights (backward compatibility)."""
        return self.get_insights()
    
    @abstractmethod
    def get_insights(self) -> list:
        """Generate data-driven insights."""
        pass
    
    @property
    def recommendations(self) -> list:
        """Return list of recommendations (backward compatibility)."""
        return self.get_recommendations()
    
    @abstractmethod
    def get_recommendations(self) -> list:
        """Generate actionable recommendations."""
        pass
    
    @property
    @abstractmethod
    def business_impact(self) -> str:
        """Return the expected business impact."""
        pass
    
    @property
    @abstractmethod
    def data_file(self) -> str:
        """Return the name of the data file this analysis uses."""
        pass
    
    @abstractmethod
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """Load and process data for this analysis."""
        pass
    
    @abstractmethod
    def create_visualization(self):
        """Create the visualization (plotly figure) for this analysis."""
        pass
