"""
Introduction Page - Meet Ron
Introduces the client and sets up the case study narrative
"""

import streamlit as st


def show_intro_page():
    """Display the introduction page for Ron's case study."""

    st.markdown("""
        <div style="
            background-color: #fff9e6;
            border-left: 5px solid #f59e0b;
            padding: 1.5rem;
            border-radius: 5px;
            margin-bottom: 2rem;
        ">
            <h3 style="margin-top: 0; color: #92400e;">About This Case Study</h3>
            <p style="color: #78350f; margin-bottom: 0;">
                <strong>What you're viewing:</strong> This is a showcase of our analytical process: 
                the "art of the possible" for data-driven decision making. In a real engagement, we would not walk a customer through each of these.
                Visualizations would be shown sparingly depending on what is most relevant for revenue growth, operational effiency gains and profit improvement.
                We tailor our communication to individual business owners: some people absorb information best with an in-depth, detailed analysis, others prefer a story-driven format, others are more visual. We can accommodate any of these preferences. 
            </p>
            </p>
            <p style="color: #78350f; margin-bottom: 0;">
                <strong>What Ron could receive:</strong> A focused 1-page executive summary with 3-5 key findings, 
                an interactive dashboard for the analyses that matter most, and a prioritized 30-day action plan. 
                We'll meet on a monthly basis to keep updating the 30-day plan, changing the relevant analyses and re-assessing performance.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Hero section
    st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        ">
            <h1 style="margin: 0; font-size: 2.5rem;">A Sample Amralytics Project: Meet Ron</h1>

        </div>
    """, unsafe_allow_html=True)
    
    # Ron's story
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### The Business Owner")
        st.markdown("""
        Ron has owned his own **HVAC business for the past 25 years**. He loves spending time with 
        clients, and it shows in his high client retention rate. However, many of the customers Ron had 
        are getting older and downsizing their homes and apartments.
        
        **Competition from big franchise HVAC companies** with fancy social media managers has made it 
        difficult to bring in new customers while still keeping an eye on margins, scheduling, outreach 
        to existing customers, and all the other tasks that pile up as a small business owner.
        """)
        
        st.markdown("### The Challenge")
        st.markdown("""
        Ron has been spending money on various marketing outreach strategies, attending networking events, 
        and trying to post on Instagram when he can. The new marketing efforts have meant he hasn't checked 
        his **prices or scheduling efficiency in a while**.
        
        **Why update prices when you don't have your next customer yet?**
        """)
    
    with col2:
        st.info("""
        **Ron's Pain Points:**
        
        1. Aging customer base downsizing
        
        2. Competition from franchises
        
        3. Struggling with social media
        
        4. Outdated pricing strategy
        
        5. Inefficient scheduling
        
        6. Overwhelmed by tasks
        """)
    
    st.divider()
    
    # Our approach
    st.markdown("### Our Goal")
    st.markdown("""
    Our goal is to **take things off Ron's to-do list, not add to it.** We'll perform a 360° analysis 
    of Ron's business, tracking how every dollar and client interaction build the picture of where things 
    are going great and where they could use some improvement.
    """)
    
    # Data sources
    st.markdown("### The Data")
    st.markdown("Ron uses several systems to run his business. We'll pull data from all of them:")
    
    cols = st.columns(5)
    with cols[0]:
        st.markdown("""
        **Operations**
        - QuickBooks (accounting)
        - ServiceTitan (scheduling)
        """)
    
    with cols[1]:
        st.markdown("""
        **Online Presence**
        - Squarespace website
        - Google Analytics
        """)
    
    with cols[2]:
        st.markdown("""
        **Marketing**
        - LinkedIn
        - Instagram
        - Google Local Services Ads
        """)
    
    with cols[3]:
        st.markdown("""
        **Reviews**
        - 100+ Yelp reviews
        - Google Business Profile
        """)
    
    with cols[4]:
        st.markdown("""
        **External Data**
        - Weather (NOAA)
        """)
    
    st.divider()
    
    # Analysis journey
    st.markdown("### The Process")
    st.markdown("""
    We completed **9 interconnected analyses**, each building on the previous to create a 
    comprehensive picture of Ron's business. We'll walk through each of these to demonstrate how they add value to Ron's business individually, and how they become even more powerful when used as building blocks to the final recommendations. Every analysis follows the same framework:
    """)
    
    journey_cols = st.columns(5)
    steps = [
        ("", "The Question", "What specific problem does Ron face?"),
        ("", "The Data", "What information did we gather?"),
        ("", "The Analysis", "Analysis results & visualizations"),
        ("", "The Insights", "Insights specific to Ron's situation"),
        ("", "The Action Plan", "Clear next steps for Ron")
    ]
    
    for col, (icon, title, desc) in zip(journey_cols, steps):
        with col:
            st.markdown(f"""
                <div style="
                    text-align: center;
                    padding: 1rem;
                    background: #f8f9fa;
                    border-radius: 8px;
                    height: 140px;
                ">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div style="font-weight: 600; margin-bottom: 0.25rem;">{title}</div>
                    <div style="font-size: 0.9rem; color: #6c757d;">{desc}</div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("")  # spacing
    
    # The 10 analyses
    st.markdown("### Our 10-Step Analysis Plan")
    
    analyses = [
        ("1️⃣", "What does Ron's business look like today?", "Business Overview - EDA & descriptive statistics"),
        ("2️⃣", "What are the characteristics of Ron's key customers?", "Customer Segmentation - Clustering analysis"),
        ("3️⃣", "How do customers feel about Ron's business?", "Sentiment Analysis - NLP on reviews"),
        ("4️⃣", "What do customers say about Ron's business?", "Text mining - Topic Extraction"),
        ("5️⃣", "How do good leads reach Ron? Which marketing channels work best?", "Marketing analytics"),
        ("6️⃣", "Which customers are at risk of leaving?", "Churn Prediction - Classification modeling"),
        ("7️⃣", "Are services priced correctly?", "Market Basket Analysis - Association Rules - Pricing Optimization"),
        ("8️⃣", "When will Ron be busiest?", "Demand Forecasting - Time series prediction"),
        ("9️⃣", "What sales patterns are normal vs. concerning", "Seasonality Analysis - Decomposition"),
        ("🔟", "Bringing it all together", "Final Synthesis - Strategic roadmap")
    ]
    
    for num, title, desc in analyses:
        st.markdown(f"""
            <div style="
                padding: 0.75rem 1rem;
                margin: 0.5rem 0;
                background: white;
                border-left: 4px solid #667eea;
                border-radius: 4px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            ">
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">{num}</span>
                <strong>{title}</strong>
                <span style="color: #6c757d; margin-left: 0.5rem;">— {desc}</span>
            </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Call to action
    st.markdown("### Ready to Begin?")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start the Analysis Journey", use_container_width=True, type="primary"):
            st.session_state.show_intro = False
            st.session_state.current_analysis_index = 0
            st.session_state.current_step = 0 
            st.rerun()
    
    st.markdown("")
    st.info("""
        **Tip:** Each analysis builds on the previous ones. We recommend following the journey in order, 
        but you can jump around using the sidebar if you prefer.
    """)
