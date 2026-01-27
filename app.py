"""
Ron's HVAC Business - 360° Data Analytics Case Study
Main Streamlit Application

A narrative-driven case study showing how data analytics helps a real small business owner.
"""

import streamlit as st
from pathlib import Path

# Import intro and synthesis pages
from intro_page import show_intro_page
from synthesis_page import show_synthesis_page

# Import all analysis modules
from analyses.business_overview import BusinessOverview
from analyses.customer_segmentation import CustomerSegmentation
from analyses.sentiment_analysis import SentimentAnalysis
from analyses.topic_extraction import TopicExtraction
from analyses.churn_modeling import ChurnModeling
from analyses.demand_forecasting import DemandForecasting
from analyses.seasonality_timeseries import SeasonalityTimeSeries
from analyses.basket_analysis import BasketAnalysis
from analyses.marketing_impact import MarketingImpact
from analyses.pricing_analysis import PricingAnalysis

# Page config
st.set_page_config(
    page_title="Ron's HVAC Business Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .insight-box {
        background-color: #f0fdf4;
        border-left: 4px solid #10b981;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .recommendation-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .impact-box {
        background-color: #fef3c7;
        border: 2px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .data-input-box {
        background-color: #eff6ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #3b82f6;
        margin-bottom: 0.5rem;
    }
    .cross-reference-box {
        background-color: #dbeafe;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .progress-bar-container {
        background-color: #e5e7eb;
        height: 8px;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .progress-bar-fill {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Force scroll to top on page load */
    .main .block-container {
        padding-top: 1rem;
    }
    
    section.main {
        scroll-behavior: smooth;
    }
    </style>
    
    <script>
        // Force scroll to top on every page render
        window.addEventListener('load', function() {
            window.scrollTo(0, 0);
        });
    </script>
""", unsafe_allow_html=True)

# Initialize session state
if 'show_intro' not in st.session_state:
    st.session_state.show_intro = True

if 'current_analysis_index' not in st.session_state:
    st.session_state.current_analysis_index = 0

if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

# ADD THIS SECTION:
if 'scroll_to_top' not in st.session_state:
    st.session_state.scroll_to_top = False

# Check if we need to scroll
if st.session_state.scroll_to_top:
    st.markdown("""
        <script>
            window.parent.document.querySelector('section.main').scrollTo(0, 0);
        </script>
    """, unsafe_allow_html=True)
    st.session_state.scroll_to_top = False  # Reset flag
# Define the analysis journey in order
# Define the analysis journey in order
ANALYSES = [
    {
        'name': 'Business Overview',
        'short_name': 'Current state of the business',
        'icon': '',
        'module': BusinessOverview(),
        'description': 'Understanding Ron\'s current state'
    },
    {
        'name': 'Customer Segmentation',
        'short_name': 'Who are Ron\'s customers?',
        'icon': '',
        'module': CustomerSegmentation(),
        'description': 'Who are Ron\'s customers?'
    },
    {
        'name': 'Sentiment Analysis',
        'short_name': 'How do customers feel?',
        'icon': '',
        'module': SentimentAnalysis(),
        'description': 'What do customers say?'
    },
    {
        'name': 'Topic Extraction',
        'short_name': 'What do customers talk about?',
        'icon': '',
        'module': TopicExtraction(),
        'description': 'Key themes in feedback'
    },
    {
        'name': 'Marketing Impact',
        'short_name': 'Which marketing channels work?',
        'icon': '',
        'module': MarketingImpact(),
        'description': 'Which channels deliver ROI?'
    },
    {
        'name': 'Churn Prediction',
        'short_name': 'Who\'s at risk of leaving?',
        'icon': '',
        'module': ChurnModeling(),
        'description': 'Who\'s at risk of leaving?'
    },
    {
        'name': 'Pricing Analysis',
        'short_name': 'Are services priced correctly?',
        'icon': '',
        'module': PricingAnalysis(),
        'description': 'Are services priced correctly?'
    },
    {
        'name': 'Demand Forecasting',
        'short_name': 'When will the business be busy?',
        'icon': '',
        'module': DemandForecasting(),
        'description': 'When will Ron be busy?'
    },
    {
        'name': 'Seasonality Analysis',
        'short_name': 'Normal vs. concerning patterns',
        'icon': '',
        'module': SeasonalityTimeSeries(),
        'description': 'Normal vs concerning patterns'
    },
    {
        'name': 'Market Basket Analysis',
        'short_name': 'What services could be bundled?',
        'icon': '',
        'module': BasketAnalysis(),
        'description': 'Service bundling opportunities'
    }
]
# Show intro page or main content
if st.session_state.show_intro:
    show_intro_page()
    
else:
    # Check if we're at synthesis
    if st.session_state.current_analysis_index >= len(ANALYSES):
        show_synthesis_page()
    
    else:
        # Main analysis view
        current_analysis = ANALYSES[st.session_state.current_analysis_index]
        analysis_module = current_analysis['module']
        
        # Sidebar navigation
        st.sidebar.markdown("## 🏠 Ron's HVAC Analysis")
        
        if st.sidebar.button("← Back to Introduction", use_container_width=True):
            st.session_state.show_intro = True
            st.rerun()
        
        st.sidebar.divider()
        
        # Progress indicator
        progress_pct = (st.session_state.current_analysis_index + 1) / (len(ANALYSES) + 1) * 100
        st.sidebar.markdown(f"""
            <div style="margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.9rem; font-weight: 600;">Progress</span>
                    <span style="font-size: 0.9rem; color: #6b7280;">
                        Step {st.session_state.current_analysis_index + 1} of {len(ANALYSES) + 1}
                    </span>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar-fill" style="width: {progress_pct}%;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown("### Analysis Journey")
        
        # Show all analyses in sidebar
        for idx, analysis in enumerate(ANALYSES):
            is_current = idx == st.session_state.current_analysis_index
            is_complete = idx < st.session_state.current_analysis_index
            
            if is_current:
                status = "→"
                style = "font-weight: 600; color: #667eea;"
            elif is_complete:
                status = "✓"
                style = "color: #10b981;"
            else:
                status = " "
                style = "color: #9ca3af;"
            
            if st.sidebar.button(
                f"{status} {idx + 1}. {analysis['icon']} {analysis['short_name']}",
                key=f"nav_{idx}",
                use_container_width=True,
                type="primary" if is_current else "secondary"
            ):
                st.session_state.current_analysis_index = idx
                st.session_state.current_step = 0
                st.rerun()
        
        # Synthesis button
        if st.sidebar.button(
            f"{'→' if st.session_state.current_analysis_index >= len(ANALYSES) else ' '} {len(ANALYSES) + 1}. Final Synthesis",
            key="nav_synthesis",
            use_container_width=True,
            type="secondary"
        ):
            st.session_state.current_analysis_index = len(ANALYSES)
            st.rerun()
        
        # Main content area
        # Header with breadcrumbs
        st.markdown(f"""
            <div style="margin-bottom: 2rem;">
                <div style="color: #6b7280; font-size: 0.9rem; margin-bottom: 0.5rem;">
                    Ron's HVAC Analysis → Analysis #{st.session_state.current_analysis_index + 1}
                </div>
                <h1 style="margin: 0; color: #1f2937;">
                    {current_analysis['icon']} {current_analysis['name']}
                </h1>
                <p style="color: #6b7280; margin-top: 0.5rem; font-size: 1.1rem;">
                    {current_analysis['description']}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # 5-step framework navigation
        steps = [
            {'name': 'The Question', 'icon': ''},
            {'name': 'The Data', 'icon': ''},
            {'name': 'The Analysis', 'icon': ''},
            {'name': 'The Insights', 'icon': ''},
            {'name': 'The Action Plan', 'icon': ''}
        ]
        
        # Step navigation
        st.markdown("### Analysis Workflow")
        cols = st.columns(len(steps))
        
        for idx, (col, step) in enumerate(zip(cols, steps)):
            with col:
                is_active = idx == st.session_state.current_step
                button_type = "primary" if is_active else "secondary"
                
                if st.button(
                    f"{step['icon']}\n{step['name']}",
                    key=f"step_{idx}",
                    use_container_width=True,
                    type=button_type
                ):
                    st.session_state.current_step = idx
                    st.rerun()
        
        st.divider()
        
        # Display content based on current step
        current_step = st.session_state.current_step
        
        # Step 1: Ron's Challenge
        if current_step == 0:
            st.markdown("## The Question")
            
            # Get the business question
            if hasattr(analysis_module, 'rons_challenge'):
                st.markdown(analysis_module.rons_challenge)
            elif hasattr(analysis_module, 'business_question'):
                st.markdown(analysis_module.business_question)
            
            st.info("**Why this matters**: Every analysis starts with understanding the specific problem Ron faces.")
        
        # Step 2: Data Collected
        elif current_step == 1:
            st.markdown("## The Data")
            
            # Get data inputs
            if hasattr(analysis_module, 'data_collected'):
                data_inputs = analysis_module.data_collected
            elif hasattr(analysis_module, 'data_inputs'):
                data_inputs = analysis_module.data_inputs
            else:
                data_inputs = []
            
            cols = st.columns(2)
            for idx, input_item in enumerate(data_inputs):
                with cols[idx % 2]:
                    # Convert markdown ** to HTML <strong>
                    html_item = input_item.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
                    
                    st.markdown(f"""
                        <div class='data-input-box'>
                            <p style='margin: 0; color: #1f2937;'>{html_item}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.info("**Data sources**: Everything comes from systems Ron already uses - no extra work for him!")
        # Step 3: What We Found
        elif current_step == 2:
            st.markdown("## The Analysis")
            
            # Show methodology
            if hasattr(analysis_module, 'methodology'):
                st.markdown(f"**Methodology:** {analysis_module.methodology}")
            elif hasattr(analysis_module, 'technical_output'):
                st.markdown(f"**Methodology:** {analysis_module.technical_output}")
            
            st.divider()
            
            # Load data and create visualization
            data_path = Path('data') / analysis_module.data_file
            
            try:
                if hasattr(analysis_module, 'load_data'):
                    analysis_module.load_data(str(data_path))
                
                fig = analysis_module.create_visualization()
                st.plotly_chart(fig, use_container_width=True)
                
                
            except FileNotFoundError:
                st.error(f"Data file not found: {data_path}")
                st.info("Make sure all CSV files are in the 'data' folder.")
            except Exception as e:
                st.error(f"Error loading visualization: {str(e)}")
                with st.expander("See error details"):
                    import traceback
                    st.code(traceback.format_exc())
        
        # Step 4: What This Means
        elif current_step == 3:
            st.markdown("## The Insights")
            
            # Get insights
            if hasattr(analysis_module, 'get_insights'):
                insights = analysis_module.get_insights()
            elif hasattr(analysis_module, 'insights'):
                insights = analysis_module.insights
            else:
                insights = []
            
            for idx, insight in enumerate(insights, 1):
                # Check if insight has cross-reference marker
                if "💡" in insight or "📊" in insight:
                    st.markdown(f"""
                        <div class='cross-reference-box'>
                            {insight}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class='insight-box'>
                            <strong>Insight {idx}:</strong> {insight}
                        </div>
                    """, unsafe_allow_html=True)
            
            st.info("**Key takeaway**: We translate complex data into clear insights specific to Ron's situation.")
        
        # Step 5: Action Plan
        elif current_step == 4:
            st.markdown("## The Action Plan")
            
            # Get recommendations
            if hasattr(analysis_module, 'get_recommendations'):
                recommendations = analysis_module.get_recommendations()
            elif hasattr(analysis_module, 'recommendations'):
                recommendations = analysis_module.recommendations
            else:
                recommendations = []
            
            for idx, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                    <div class='recommendation-box'>
                        <strong>Action {idx}:</strong> {rec}
                    </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Business impact
            if hasattr(analysis_module, 'business_impact'):
                st.markdown(f"""
                    <div class='impact-box'>
                        <strong>Expected Business Impact:</strong><br>
                        {analysis_module.business_impact}
                    </div>
                """, unsafe_allow_html=True)
            
            st.success("**Next**: This analysis sets up the next step in our journey!")
        
        # Navigation buttons at bottom
        st.divider()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.session_state.current_analysis_index > 0 or st.session_state.current_step > 0:
                if st.button("⬅️ Previous", use_container_width=True, type="secondary"):
                    if st.session_state.current_step > 0:
                        st.session_state.current_step -= 1
                    else:
                        st.session_state.current_analysis_index -= 1
                        st.session_state.current_step = 4  # Go to last step of previous analysis
                    st.rerun()
        
        with col3:
            # Determine next button text
            if st.session_state.current_step < 4:
                next_text = "Next Step ➡️"
            elif st.session_state.current_analysis_index < len(ANALYSES) - 1:
                next_text = "Next Analysis ➡️"
            else:
                next_text = "Final Synthesis ➡️"
            
            if st.button(next_text, use_container_width=True, type="primary"):
                if st.session_state.current_step < 4:
                    st.session_state.current_step += 1
                else:
                    st.session_state.current_analysis_index += 1
                    st.session_state.current_step = 0
                st.rerun()
