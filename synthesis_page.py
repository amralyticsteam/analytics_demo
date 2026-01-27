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
            <h1 style="margin: 0; font-size: 2.5rem;">The Complete Picture: </h1>
            <p style="font-size: 1.3rem; margin-top: 0.5rem; opacity: 0.95;">
                Combining all 9 analyses into actionable, prioritized insights for Ron
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### How Everything Connects")
    
    st.markdown("""
    We've completed a comprehensive analysis of Ron's HVAC business. Each analysis built on the previous one 
    to create a complete picture, a view of Ron's business that would be incomplete if we only
    looked at a single area of the business. Here's how they connect:
    """)
    
    # Connection flow
    st.markdown("""
    1. Business Overview: Revealed declining revenue & aging customer base
                
    ↓
       
    2. Customer Segmentation: Identified distinct groups with different needs
                             
    ↓
       
    3. Sentiment Analysis:  Showed what each segment values most
                             
    ↓
       
    4. Marketing Analysis: Found which channels reach which segments
                             
    ↓
       
    5. Churn Prediction: Identified at-risk customers in each segment
                             
    ↓
       
    6. Pricing Analysis: Showed some services underpriced, others overpriced
                             
    ↓
       
    7. Demand Forecasting: Predicted busy periods requiring staff/inventory
                             
    ↓
       
    8. Seasonality Analysis: Separated normal variation from real problems
                             
    ↓
       
    9. Complete Action Plan
    """)
    
    st.divider()
    
    # Key findings
    st.markdown("### Top 5 Key Findings")
    
    findings = [
        {
            "finding": "Ron's customer base IS aging and shrinking",
            "evidence": "Avg age 62, revenue down 8% in 6 months",
            "action": "Need to attract younger homeowners"
        },
        {
            "finding": "High-value installation jobs are rare",
            "evidence": "Only 45/year at $2,667 each",
            "action": "Marketing should emphasize installation capabilities"
        },
        {
            "finding": "Customers love Ron's reliability but complain about response times",
            "evidence": "92% positive on quality, but 'slow response' in 23% of negative reviews",
            "action": "Hire assistant or optimize scheduling"
        },
        {
            "finding": "Instagram isn't working for Ron's customer demographics",
            "evidence": "Avg customer is 62, Instagram users skew younger",
            "action": "Shift budget to Google Local Services Ads"
        },
        {
            "finding": "Pricing hasn't kept up with costs",
            "evidence": "Last price update 18 months ago, margins shrinking",
            "action": "Implement 12% price increase on maintenance"
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
                <div style="color: #6b7280; margin-bottom: 0.5rem;">
                    <strong>Evidence:</strong> {item['evidence']}
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
            "action": "Raise prices on maintenance services by 12%",
            "timeline": "This week",
            "impact": "High",
            "effort": "Low",
            "expected_result": "+$9,600 annual revenue with minimal customer loss"
        },
        {
            "priority": "🟠 Priority 2", 
            "action": "Win-back campaign for at-risk 'Golden Years' segment (250 customers)",
            "timeline": "Next 2 weeks",
            "impact": "High",
            "effort": "Medium",
            "expected_result": "Prevent $15K in lost annual revenue"
        },
        {
            "priority": "🟡 Priority 3",
            "action": "Shift marketing: Stop Instagram, increase Google Local Services Ads budget",
            "timeline": "Next month",
            "impact": "Medium",
            "effort": "Low",
            "expected_result": "Better ROI on marketing spend, 20% more qualified leads"
        },
        {
            "priority": "🟢 Priority 4",
            "action": "Hire part-time scheduler/assistant for peak season (May-August)",
            "timeline": "Before May 1",
            "impact": "Medium",
            "effort": "High",
            "expected_result": "Handle 15% more jobs, improve response time complaints"
        },
        {
            "priority": "🔵 Priority 5",
            "action": "Create installation-focused landing page + Google Ads campaign",
            "timeline": "Next 60 days",
            "impact": "High",
            "effort": "Medium",
            "expected_result": "Double installation jobs from 45 to 90/year"
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
           - Explore any analysis deeper, build data literacy over time
           - Filter by customer segment
           - See updated data daily
           
        3. **Monthly Check-ins** (30 min)
           - Review progress on action items
           - Adjust priorities as needed
        """)
    
    with col2:
        st.markdown("#### What Ron Does NOT Get")
        st.markdown("""
        - Raw data dumps
        - Technical jargon
        - 50-page PowerPoint
        - Generic industry benchmarks
        - Analyses without clear next steps
        
        *Our goal: Take things off Ron's plate, not add more work.*
        """)
    
    st.divider()
    
    # Projected impact
    st.markdown("### Projected Impact (Year 1)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Revenue Increase",
            "+$47K",
            "+8.2%",
            help="From pricing, installations, and retention"
        )
    
    with col2:
        st.metric(
            "Time Saved",
            "6 hrs/week",
            delta="→ $15K value",
            help="Better scheduling + marketing efficiency"
        )
    
    with col3:
        st.metric(
            "Customer Retention",
            "+12%",
            "35 customers",
            help="From targeted win-back campaigns"
        )
    
    st.divider()
    
    # Next steps
    st.markdown("### Next Steps")
    
    st.info("""
    **For Ron:**
    1. Review this action plan
    2. Pick Priority 1-3 to start (don't do everything at once!)
    3. Schedule 30-min implementation call
    
    **For Our Team:**
    1. Set up automated monthly reporting
    2. Create templates for customer outreach
    3. Monitor results and adjust recommendations
    """)
    
    st.divider()
    
    # Call to action
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("← Back to Introduction", use_container_width=True, type="secondary"):
            st.session_state.show_intro = True
            st.rerun()
        