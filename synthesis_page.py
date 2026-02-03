"""
Final Synthesis Page - The Complete Picture for Ron
Brings all analyses together with prioritized action plan
"""

import streamlit as st


def show_synthesis_page():
    """Display the synthesis/conclusion page."""
    
    # Hero section
    st.markdown("""
        <div style="
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            padding: 3rem 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        ">
            <h1 style="margin: 0; font-size: 2.5rem;">The Complete Picture</h1>
            <p style="font-size: 1.3rem; margin-top: 0.5rem; opacity: 0.95;">
                Combining all 10 analyses into actionable, prioritized insights for Ron
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### How Everything Connects")
    
    st.markdown("""
    We've completed a comprehensive analysis of Ron's HVAC business. Each analysis built on the previous one 
    to create a complete picture. Here's how they connect:
    """)
    
    # Visual snake flowchart
    st.markdown("""
<div style="max-width: 1200px; margin: 2rem auto;">
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1rem;">
        <div class="flow-box">
            <div class="flow-number">1</div>
            <div class="flow-title">Business Overview</div>
            <div class="flow-insight">Revenue trends</div>
            <div class="flow-source">ServiceTitan</div>
            <div class="flow-arrow">→</div>
        </div>
        <div class="flow-box">
            <div class="flow-number">2</div>
            <div class="flow-title">Customer Segmentation</div>
            <div class="flow-insight">5 customer groups</div>
            <div class="flow-source">ServiceTitan</div>
            <div class="flow-arrow">→</div>
        </div>
        <div class="flow-box">
            <div class="flow-number">3</div>
            <div class="flow-title">Sentiment Analysis</div>
            <div class="flow-insight">Customer values</div>
            <div class="flow-source">Google Reviews</div>
            <div class="flow-arrow">↓</div>
        </div>
    </div>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1rem;">
        <div class="flow-box">
            <div class="flow-number">6</div>
            <div class="flow-title">Churn Prediction</div>
            <div class="flow-insight">At-risk customers</div>
            <div class="flow-source">ServiceTitan</div>
            <div class="flow-arrow">↓</div>
        </div>
        <div class="flow-box">
            <div class="flow-number">5</div>
            <div class="flow-title">Marketing Impact</div>
            <div class="flow-insight">ROI by channel</div>
            <div class="flow-source">Google Analytics</div>
            <div class="flow-arrow">←</div>
        </div>
        <div class="flow-box">
            <div class="flow-number">4</div>
            <div class="flow-title">Topic Extraction</div>
            <div class="flow-insight">Key themes</div>
            <div class="flow-source">Google Reviews</div>
            <div class="flow-arrow">←</div>
        </div>
    </div>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1rem;">
        <div class="flow-box">
            <div class="flow-number">7</div>
            <div class="flow-title">Pricing Analysis</div>
            <div class="flow-insight">Negative margins</div>
            <div class="flow-source">QuickBooks</div>
            <div class="flow-arrow">→</div>
        </div>
        <div class="flow-box">
            <div class="flow-number">8</div>
            <div class="flow-title">Demand Forecasting</div>
            <div class="flow-insight">Predict busy periods</div>
            <div class="flow-source">ServiceTitan</div>
            <div class="flow-arrow">→</div>
        </div>
        <div class="flow-box">
            <div class="flow-number">9</div>
            <div class="flow-title">Seasonality Analysis</div>
            <div class="flow-insight">Normal vs concerning</div>
            <div class="flow-source">QuickBooks</div>
            <div class="flow-arrow">↓</div>
        </div>
    </div>
    <div style="display: flex; justify-content: center; margin-bottom: 1rem;">
        <div class="flow-box" style="width: 32%;">
            <div class="flow-number">10</div>
            <div class="flow-title">Basket Analysis</div>
            <div class="flow-insight">Bundling opportunities</div>
            <div class="flow-source">ServiceTitan</div>
            <div class="flow-arrow">↓</div>
        </div>
    </div>
    <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 2rem; border-radius: 10px; text-align: center; color: white; margin-top: 1rem;">
        <div style="font-size: 1.8rem; font-weight: bold;">Complete Action Plan</div>
        <div style="font-size: 1.1rem; margin-top: 0.5rem; opacity: 0.95;">Prioritized recommendations from all 10 analyses</div>
    </div>
</div>
<style>
.flow-box {
    background: white;
    border: 2px solid #667eea;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: transform 0.2s, box-shadow 0.2s;
    min-height: 150px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.flow-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(102, 126, 234, 0.2);
}
.flow-number {
    background: #667eea;
    color: white;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 0.5rem;
    font-weight: bold;
    font-size: 1rem;
}
.flow-title {
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 0.25rem;
    font-size: 0.95rem;
}
.flow-insight {
    font-size: 0.8rem;
    color: #6b7280;
    margin-bottom: 0.25rem;
}
.flow-source {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-bottom: 0.5rem;
}
.flow-arrow {
    font-size: 1.5rem;
    color: #667eea;
    margin-top: 0.25rem;
    font-weight: bold;
    line-height: 1;
}
.flow-arrow-top {
    font-size: 1.5rem;
    color: #667eea;
    margin-bottom: 0.25rem;
    font-weight: bold;
    line-height: 1;
}
</style>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Key findings - Updated with real data from analyses
    st.markdown("### Top 5 Key Findings")
    
    findings = [
        {
            "finding": "Ron's most popular services are losing money",
            "evidence": "Thermostat Installation (156 jobs, -1.9% margin), Smart Thermostat (98 jobs, -6.8% margin), Heating Tune-up (92 jobs, -5.4% margin) - the top 3 most popular services all have negative margins. 7 services total losing money.",
            "source": "Pricing Analysis - QuickBooks & ServiceTitan",
            "action": "Immediate 15-20% price increase on these 7 services"
        },
        {
            "finding": "Customer segments have very different needs",
            "evidence": "VIP customers (10%) average $3,500 per service vs Occasional customers (35%) at $400. 650 customers across 5 distinct RFM segments.",
            "source": "Customer Segmentation - ServiceTitan",
            "action": "Tailor marketing & service packages by segment"
        },
        {
            "finding": "High churn risk among recent customers",
            "evidence": "99 customers (15.2%) flagged as high churn risk. Customers inactive 12+ months or with low engagement.",
            "source": "Churn Prediction - ServiceTitan",
            "action": "Launch win-back campaign immediately"
        },
        {
            "finding": "Service bundling opportunities untapped",
            "evidence": "Strong associations found: Thermostat Installation frequently paired with other services (1.5-2.5x lift)",
            "source": "Basket Analysis - ServiceTitan invoices",
            "action": "Create service packages with 10-15% bundle discount"
        },
        {
            "finding": "Marketing channels have wildly different ROI",
            "evidence": "Some channels deliver 3-5x ROAS, others lose money on every campaign",
            "source": "Marketing Impact - Google Analytics & Meta Ads",
            "action": "Reallocate budget to high-performing channels"
        }
    ]
    
    for i, item in enumerate(findings, 1):
        st.markdown(f"""
            <div style="
                background: white;
                padding: 1.5rem;
                margin: 1rem 0;
                border-radius: 8px;
                border-left: 5px solid #667eea;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <div style="font-size: 1.2rem; font-weight: 600; color: #667eea; margin-bottom: 0.5rem;">
                    {i}. {item['finding']}
                </div>
                <div style="color: #6b7280; margin-bottom: 0.3rem;">
                    <strong>Evidence:</strong> {item['evidence']}
                </div>
                <div style="color: #9ca3af; font-size: 0.9rem; margin-bottom: 0.5rem;">
                    {item['source']}
                </div>
                <div style="color: #059669; font-weight: 500;">
                    <strong>→ Action:</strong> {item['action']}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Prioritized action plan
    st.markdown("### 90-Day Action Plan (Prioritized)")
    
    st.markdown("""
    We've identified 15+ possible improvements. Here are the **Top 5 to tackle first**, ordered by:
    - **Impact**: How much will this move the needle?
    - **Effort**: How hard is it to implement?
    - **Timeline**: How fast can Ron see results?
    """)
    
    actions = [
        {
            "priority": "🔴 Priority 1",
            "action": "Fix pricing on 7 negative-margin services immediately",
            "timeline": "This week",
            "impact": "High",
            "effort": "Low",
            "expected_result": "Raise Thermostat Install from \$189 to \$225, Smart Thermostat from \$295 to \$350, Heating Tune-up from \$140 to \$165 (+ 4 others). Stop losing money on 346 annual jobs. +\$18-25K annual profit.",
            "source": "Pricing Analysis"
        },
        {
            "priority": "🟠 Priority 2", 
            "action": "Win-back campaign for high-risk churn customers",
            "timeline": "Next 2 weeks",
            "impact": "High",
            "effort": "Medium",
            "expected_result": "Email/call 99 high-risk customers. 'We Miss You' seasonal tune-up offer. Target 25% win-back rate = ~$50K preserved revenue.",
            "source": "Churn Prediction"
        },
        {
            "priority": "🟡 Priority 3",
            "action": "Create service bundles based on association rules",
            "timeline": "Next month",
            "impact": "High",
            "effort": "Medium",
            "expected_result": "Package frequently co-purchased services at 10\% discount. Increase average ticket 15-25%.",
            "source": "Basket Analysis"
        },
        {
            "priority": "🟢 Priority 4",
            "action": "Reallocate marketing spend to high-ROI channels",
            "timeline": "Next 30 days",
            "impact": "Medium-High",
            "effort": "Low",
            "expected_result": "Cut spending on low-ROAS channels, double down on Google Local Services. 3-5x better ROI.",
            "source": "Marketing Impact"
        },
        {
            "priority": "🔵 Priority 5",
            "action": "Staff up for predicted busy season (demand forecast)",
            "timeline": "Before peak season",
            "impact": "Medium",
            "effort": "High",
            "expected_result": "Hire temp help for high-demand periods. Handle 20% more calls without quality drop.",
            "source": "Demand Forecasting"
        }
    ]
    
    for item in actions:
        with st.expander(f"{item['priority']}: {item['action']}", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Timeline", item['timeline'])
            with col2:
                st.metric("Impact", item['impact'])
            with col3:
                st.metric("Effort", item['effort'])
            
            st.success(f"**Expected Result:** {item['expected_result']}")
            st.info(f"**Data Source:** {item['source']}")
    
    st.divider()
    
    # Data sources used
    st.markdown("### Data Sources - Everything From Ron's Existing Systems")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔧 ServiceTitan** (Field Service Software)
        - 3,490 service transactions
        - 650 unique customers
        - 2,358 invoices with service bundles
        - Customer history & churn indicators
        - Daily call volume (336 days)
        
        **💰 QuickBooks** (Accounting)
        - 50 services with pricing & COGS
        - 32 months of revenue data
        - Parts costs & margins
        """)
    
    with col2:
        st.markdown("""
        **🌐 Google Reviews** (Customer Feedback)
        - 40 customer reviews
        - Ratings, sentiment, topics
        
        **📊 Marketing Platforms**
        - Google Analytics
        - Meta Ads Manager
        - 37 campaigns across 6 channels
        
        **🌤️ Weather API** (NOAA)
        - Temperature, precipitation data
        - Correlated with demand patterns
        """)
    
    st.info("**No extra work for Ron** - we pulled data from systems he already uses daily!")
    
    st.divider()
    
    # How Ron receives this
    st.markdown("### Delivering Insights to Ron")
    
    st.markdown("""
    Ron is a **hands-on business owner, not a data analyst.** We learned in our initial conversation that he:
    - Prefers clear, visual summaries over spreadsheets
    - Wants to understand the "why" behind recommendations
    - Needs actionable steps, not theoretical insights
    - Values his time - keep it concise
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### What Ron Gets")
        st.markdown("""
        1. **Executive Summary** (1 page)
           - Top 5 findings
           - 7-day, 30-day, 90-day action plan
           
        2. **Interactive Dashboard**
           - Explore any analysis deeper
           - Filter by customer segment
           - See updated data monthly
           
        3. **Monthly Check-ins** (30 min)
           - Review progress on action items
           - Adjust priorities as needed
           - No surprises, collaborative approach
        """)
    
    with col2:
        st.markdown("#### What Ron Does NOT Get")
        st.markdown("""
        - Raw data dumps
        - Technical jargon
        - 50-page PowerPoint
        - Generic industry benchmarks
        - Analyses without clear next steps
        - Synthetic/fake data
        
        *Our goal: Take things off Ron's plate, not add more work.*
        """)
    
    st.divider()
    
    # Projected impact - Updated with realistic numbers
    st.markdown("### Projected Impact (Year 1)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Revenue Increase",
            "+$25-30K",
            "+2-3%",
            help="From pricing fixes on 7 negative-margin services"
        )
    
    with col2:
        st.metric(
            "Margin Improvement",
            "+12-18%",
            "On fixed services",
            help="Move from -7% to +15% average on problem services"
        )
    
    with col3:
        st.metric(
            "Customer Retention",
            "+15-20%",
            "Win-back campaigns",
            help="Prevent churn in at-risk segments"
        )
    
    with col4:
        st.metric(
            "Marketing ROI",
            "3-5x",
            "Better targeting",
            help="Reallocate to high-performing channels"
        )
    
    st.divider()
    
    # The Amralytics difference
    st.markdown("### The Amralytics Approach")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Comprehensive**
        
        Not just one analysis - we look at the entire business ecosystem and how everything connects.
        """)
    
    with col2:
        st.markdown("""
        **🎯 Actionable**
        
        Every insight comes with specific next steps, timelines, and expected results. No theory, just action.
        """)
    
    with col3:
        st.markdown("""
        **🤝 Collaborative**
        
        We work with Ron's existing systems and workflows. No disruption, just improvement.
        """)
    
    st.divider()
    
    # Call to action
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("← Back to Introduction", use_container_width=True, type="secondary"):
            st.session_state.show_intro = True
            st.session_state.current_analysis_index = 0
            st.session_state.current_step = 0
            st.rerun()
