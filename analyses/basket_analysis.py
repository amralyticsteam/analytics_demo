"""Market Basket Analysis Module - Amralytics Methodology"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from .base_analysis import BaseAnalysis

# Association rule mining
try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    print("Note: mlxtend not available. Install with: pip install mlxtend")


class BasketAnalysis(BaseAnalysis):
    """Analyze HVAC service associations and bundling opportunities using association rules."""
    
    def __init__(self):
        super().__init__()
        self.basket_df = None
        self.transactions_df = None
        self.basket_encoded = None
        self.freq_itemsets = None
        self.rules = None
        self.top_pairs = None
    
    @property
    def icon(self) -> str:
        return '🛒'
    
    @property
    def color(self) -> str:
        return '#23606e'
    
    @property
    def rons_challenge(self) -> str:
        return """When a customer calls for an AC repair, what else do they typically need? When Ron 
        installs a new furnace, which add-on services should he recommend?
        
**Understanding service bundling patterns** helps Ron create packages, train technicians on 
upselling, and increase average ticket size without being pushy."""
    
    # Backward compatibility
    @property
    def business_question(self) -> str:
        return self.rons_challenge
    
    @property
    def data_collected(self) -> list:
        return [
            'Invoice data (1,000 invoices) - **ServiceTitan**',
            'Bundled services per visit - **ServiceTitan**',
            'Service co-occurrence patterns - **ServiceTitan**',
            'Same-visit service combinations - **ServiceTitan**',
            'Customer and date information - **ServiceTitan**'
    ]
    
    # Backward compatibility
    @property
    def data_inputs(self) -> list:
        return self.data_collected
    
    @property
    def methodology(self) -> str:
        return """We use the following analytical techniques to discover which services are naturally purchased together:

**Market Basket Analysis (Apriori Algorithm)** - A machine learning technique from retail (think "people who buy beer also buy diapers") adapted for services. Finds rules like "customers who get Thermostat Installation also get Furnace Maintenance 75% of the time."

**Association rules (Support, Confidence, Lift)** - Metrics that measure how strong the connection is between services. Lift > 2.0 means the combo happens twice as often as random chance.

**Network visualization** - Shows services as nodes with connections between frequently bundled items, revealing the ecosystem of related services.

**Phi correlation** - Statistical measure of how strongly services co-occur on the same invoice.

**Why this works for Ron:** Creates data-driven service bundles and trains technicians on natural upsell opportunities ("Since we're installing your furnace, let me also check your thermostat...").

**If results aren't strong enough, we could:**
- Add sequence mining (what do customers buy first, second, third over time?)
- Include seasonal basket patterns (summer vs winter service combos)
- Build recommendation engine (Netflix-style "customers who bought X also bought Y")
- Analyze basket profitability (not just popularity)"""
    # Backward compatibility
    @property
    def technical_output(self) -> str:
        return self.methodology
    
    @property
    def data_file(self) -> str:
        return 'basket_analysis.csv'
    
    def build_transactions(self, df):
        """Convert invoice data to transaction format for association rules."""
        service_cols = ["service_1", "service_2", "service_3"]
        
        # Melt to long format
        long = (
            df[["invoice_id", "customer_id", "date", "total_amount", "same_visit"] + service_cols]
            .melt(
                id_vars=["invoice_id", "customer_id", "date", "total_amount", "same_visit"],
                value_vars=service_cols,
                value_name="service"
            )
            .drop(columns="variable")
        )
        
        # Clean service names
        long["service"] = (
            long["service"]
            .astype(str)
            .str.strip()
            .replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan})
        )
        long = long.dropna(subset=["service"])
        
        # Group by invoice to get list of services
        transactions_df = (
            long.groupby("invoice_id")
            .agg(
                services=("service", lambda s: sorted(set(s))),
                customer_id=("customer_id", "first"),
                date=("date", "first"),
                total_amount=("total_amount", "first"),
                same_visit=("same_visit", "first"),
            )
            .reset_index()
        )
        
        return transactions_df
    
    def run_association_rules(self, transactions_df, min_support=0.01, min_confidence=0.15, min_lift=1.1):
        """Run Apriori algorithm and generate association rules."""
        if not MLXTEND_AVAILABLE:
            return None, None
        
        transactions = transactions_df["services"].tolist()
        n_tx = len(transactions)
        
        # Encode transactions
        te = TransactionEncoder()
        te_data = te.fit(transactions).transform(transactions)
        basket = pd.DataFrame(te_data, columns=te.columns_)
        
        # Remove ultra-rare services (noise reduction)
        min_item_count = max(5, int(0.005 * n_tx))
        item_counts = basket.sum(axis=0)
        keep_cols = item_counts[item_counts >= min_item_count].index
        basket = basket[keep_cols]
        
        # Run Apriori
        freq = apriori(basket, min_support=min_support, use_colnames=True)
        freq["length"] = freq["itemsets"].apply(len)
        
        # Generate rules
        if len(freq[freq["length"] >= 2]) == 0:
            return basket, pd.DataFrame()
        
        rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
        
        if rules.empty:
            return basket, rules
        
        # Filter by lift and remove trivial rules
        rules = rules[rules["lift"] >= min_lift].copy()
        
        # Create readable rule strings
        rules["antecedent"] = rules["antecedents"].apply(lambda x: next(iter(x)) if len(x) == 1 else str(x))
        rules["consequent"] = rules["consequents"].apply(lambda x: next(iter(x)) if len(x) == 1 else str(x))
        rules["rule"] = rules["antecedent"] + " → " + rules["consequent"]
        
        # Sort by lift
        rules = rules.sort_values("lift", ascending=False)
        
        return basket, rules
    
    def calculate_top_pairs(self, basket):
        """Calculate top correlated service pairs using phi coefficient."""
        if basket is None or basket.empty:
            return pd.DataFrame()
        
        item_support = basket.mean()
        keep = item_support[item_support >= 0.02].index  # Keep items with 2%+ support
        X = basket[keep].astype(int)
        
        if X.shape[1] < 2:
            return pd.DataFrame()
        
        corr = X.corr()  # Phi coefficient for binary variables
        
        # Extract upper triangle
        pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if not pd.isna(corr.iat[i, j]):
                    pairs.append({
                        'service_a': cols[i],
                        'service_b': cols[j],
                        'correlation': corr.iat[i, j]
                    })
        
        out = pd.DataFrame(pairs)
        if len(out) > 0:
            out = out.sort_values('correlation', ascending=False).head(15)
        
        return out
    
    def abbreviate_service_name(self, service_name, max_length=25):
        """Abbreviate service names for better visualization display."""
        # Common abbreviations
        abbrev_map = {
            'Installation': 'Install',
            'Replacement': 'Replace',
            'Inspection': 'Inspect',
            'Cleaning': 'Clean',
            'Thermostat': 'Tstat',
            'Refrigerant': 'Refrig',
            'Condenser': 'Cond',
            'Evaporator': 'Evap',
            'Compressor': 'Comp',
            'Heat Exchanger': 'HX',
            'Air Handler': 'AH',
            'Mini-Split': 'Mini',
        }
        
        # Apply abbreviations
        abbreviated = service_name
        for full, abbrev in abbrev_map.items():
            abbreviated = abbreviated.replace(full, abbrev)
        
        # If still too long, truncate with ellipsis
        if len(abbreviated) > max_length:
            abbreviated = abbreviated[:max_length-3] + '...'
        
        return abbreviated
    
    def format_rule_text(self, rule_str):
        """Format rule text with line breaks for better display."""
        if ' → ' in rule_str:
            parts = rule_str.split(' → ')
            if len(parts) == 2:
                # Abbreviate both parts
                ant = self.abbreviate_service_name(parts[0], max_length=20)
                cons = self.abbreviate_service_name(parts[1], max_length=20)
                return f"{ant}<br>→ {cons}"
        return self.abbreviate_service_name(rule_str, max_length=25)
    
    def format_pair_text(self, service_a, service_b):
        """Format service pair text with line break."""
        a_short = self.abbreviate_service_name(service_a, max_length=15)
        b_short = self.abbreviate_service_name(service_b, max_length=15)
        return f"{a_short}<br>+ {b_short}"
    
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """Load and process basket analysis data."""
        if filepath is None:
            filepath = f'data/{self.data_file}'
        
        print(f"Loading basket data from: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} invoices")
        
        # Build transactions
        self.transactions_df = self.build_transactions(df)
        print(f"Created {len(self.transactions_df)} transactions")
        
        # Run association rules
        if MLXTEND_AVAILABLE:
            self.basket_encoded, self.rules = self.run_association_rules(
                self.transactions_df,
                min_support=0.01,
                min_confidence=0.15,
                min_lift=1.1
            )
            
            if self.rules is not None and len(self.rules) > 0:
                print(f"Found {len(self.rules)} association rules")
            else:
                print("No strong association rules found")
            
            # Calculate correlations
            self.top_pairs = self.calculate_top_pairs(self.basket_encoded)
        else:
            print("mlxtend not available - skipping association rules")
        
        self.basket_df = df
        self.data = df
        return df
    
    def create_visualization(self):
        """Create 4-panel basket analysis dashboard."""
        if self.basket_df is None:
            self.load_data()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Association Rules: Support vs Confidence',
                'Top Service Pairs by Lift',
                'Network: Strong Service Associations',
                'Service Co-occurrence Patterns'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ],
            vertical_spacing=0.22,
            horizontal_spacing=0.15
        )
        
        if self.rules is not None and len(self.rules) > 0:
            # 1. Support vs Confidence Scatter (bubble size = lift)
            fig.add_trace(
                go.Scatter(
                    x=self.rules['support'],
                    y=self.rules['confidence'],
                    mode='markers',
                    name='Association Rules',
                    marker=dict(
                        size=self.rules['lift'] * 15,
                        color=self.rules['lift'],
                        colorscale=[[0, '#ffa07a'], [0.5, '#008f8c'], [1, '#00b894']],
                        showscale=True,
                        colorbar=dict(title="Lift", x=0.46, len=0.4),
                        line=dict(width=1, color='white'),
                        sizemode='diameter',
                        sizemin=5
                    ),
                    text=self.rules['rule'],
                    hovertemplate=(
                        '<b>%{text}</b><br>'
                        'Support: %{x:.3f}<br>'
                        'Confidence: %{y:.3f}<br>'
                        'Lift: %{marker.color:.2f}<extra></extra>'
                    ),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # 2. Top Pairs by Lift - Horizontal bars
            top_rules = self.rules.nlargest(10, 'lift')
            
            # Format rule text with abbreviations and line breaks
            formatted_rules = [self.format_rule_text(rule) for rule in top_rules['rule']]
            
            colors_lift = []
            for lift in top_rules['lift']:
                if lift >= 3:
                    colors_lift.append('#00b894')
                elif lift >= 2:
                    colors_lift.append('#008f8c')
                else:
                    colors_lift.append('#23606e')
            
            fig.add_trace(
                go.Bar(
                    y=formatted_rules,
                    x=top_rules['lift'],
                    orientation='h',
                    marker_color=colors_lift,
                    text=[f"{l:.2f}x" for l in top_rules['lift']],
                    textposition='outside',
                    showlegend=False,
                    hovertemplate='<b>%{customdata}</b><br>Lift: %{x:.2f}x<extra></extra>',
                    customdata=top_rules['rule']  # Show full rule on hover
                ),
                row=1, col=2
            )
            
            # 3. Network Visualization - Top 15 rules
            top_15 = self.rules.nlargest(15, 'lift')
            
            G = nx.DiGraph()
            node_abbrev = {}  # Map full name to abbreviation
            
            for _, row in top_15.iterrows():
                a = row['antecedent']
                b = row['consequent']
                
                # Create abbreviations for network nodes
                if a not in node_abbrev:
                    node_abbrev[a] = self.abbreviate_service_name(a, max_length=15)
                if b not in node_abbrev:
                    node_abbrev[b] = self.abbreviate_service_name(b, max_length=15)
                
                G.add_edge(a, b, lift=float(row['lift']), conf=float(row['confidence']))
            
            if G.number_of_nodes() > 0:
                pos = nx.spring_layout(G, seed=42, k=0.8)
                
                # Edges
                edge_x, edge_y = [], []
                for u, v in G.edges():
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]
                
                fig.add_trace(
                    go.Scatter(
                        x=edge_x, y=edge_y,
                        mode='lines',
                        line=dict(width=1.5, color='#98d8c8'),
                        hoverinfo='none',
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
                # Nodes - use abbreviated names for display
                node_x, node_y, node_text, node_hover, node_sizes = [], [], [], [], []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node_abbrev[node])  # Abbreviated for display
                    node_hover.append(node)  # Full name on hover
                    # Size by degree (how many connections)
                    node_sizes.append(15 + G.degree(node) * 10)
                
                fig.add_trace(
                    go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=node_text,
                        textposition='top center',
                        textfont=dict(size=8),
                        marker=dict(
                            size=node_sizes,
                            color='#008f8c',
                            line=dict(width=2, color='white')
                        ),
                        hovertemplate='<b>%{customdata}</b><extra></extra>',
                        customdata=node_hover,
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
                # Hide axes for network
                fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=2, col=1)
                fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=2, col=1)
        
        # 4. Service Co-occurrence (from correlations)
        if self.top_pairs is not None and len(self.top_pairs) > 0:
            top_corr = self.top_pairs.head(12)
            
            # Format pair names with line breaks
            formatted_pairs = [
                self.format_pair_text(row['service_a'], row['service_b']) 
                for _, row in top_corr.iterrows()
            ]
            
            # Store full names for hover
            full_pairs = [
                f"{row['service_a']} + {row['service_b']}" 
                for _, row in top_corr.iterrows()
            ]
            
            colors_corr = []
            for corr in top_corr['correlation']:
                if corr >= 0.5:
                    colors_corr.append('#00b894')
                elif corr >= 0.3:
                    colors_corr.append('#008f8c')
                else:
                    colors_corr.append('#23606e')
            
            fig.add_trace(
                go.Bar(
                    y=formatted_pairs,
                    x=top_corr['correlation'],
                    orientation='h',
                    marker_color=colors_corr,
                    text=[f"{c:.2f}" for c in top_corr['correlation']],
                    textposition='outside',
                    showlegend=False,
                    hovertemplate='<b>%{customdata}</b><br>Correlation: %{x:.3f}<extra></extra>',
                    customdata=full_pairs
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=False,
            title_text="Market Basket Analysis: Service Bundling Opportunities",
            title_x=0.5,
            title_font=dict(size=18, color='#023535'),
            margin=dict(l=150, r=80, t=100, b=80)  # Increased left margin for labels
        )
        
        # Update axes
        fig.update_xaxes(title_text="Support (% of Transactions)", row=1, col=1)
        fig.update_yaxes(title_text="Confidence", row=1, col=1)
        
        fig.update_xaxes(title_text="Lift (Strength of Association)", row=1, col=2)
        fig.update_yaxes(title_text="Service Pair", row=1, col=2)
        
        fig.update_xaxes(title_text="Correlation Coefficient", row=2, col=2)
        fig.update_yaxes(title_text="Service Pair", row=2, col=2)
        
        return fig
    
    def get_insights(self) -> list:
        """Generate data-driven insights from basket analysis."""
        if self.basket_df is None:
            self.load_data()
        
        insights = []
        
        insights.append(f"Analyzed {len(self.basket_df)} invoices to identify service bundling patterns")
        
        if self.rules is not None and len(self.rules) > 0:
            # Strongest rule
            top_rule = self.rules.iloc[0]
            insights.append(
                f"**Strongest association**: {top_rule['rule']} "
                f"(lift: {top_rule['lift']:.2f}x, confidence: {top_rule['confidence']*100:.0f}%)"
            )
            
            # High confidence rules
            high_conf = self.rules[self.rules['confidence'] > 0.5]
            if len(high_conf) > 0:
                insights.append(
                    f"{len(high_conf)} service pairs have >50% co-purchase rate - "
                    f"strong bundling opportunities"
                )
            
            # Installation patterns
            install_rules = self.rules[
                self.rules['antecedent'].str.contains('Installation', case=False, na=False)
            ]
            if len(install_rules) > 0:
                top_install = install_rules.iloc[0]
                insights.append(
                    f"Installation insight: {top_install['rule']} "
                    f"- always recommend {top_install['consequent']} with installations"
                )
        else:
            insights.append("No strong association rules found - services are typically standalone")
        
        # Correlation insights
        if self.top_pairs is not None and len(self.top_pairs) > 0:
            top_pair = self.top_pairs.iloc[0]
            insights.append(
                f"Most correlated services: {top_pair['service_a']} + {top_pair['service_b']} "
                f"(correlation: {top_pair['correlation']:.2f})"
            )
        
        # Connection to other analyses
        insights.append(
            "**Connection to Pricing Analysis**: Bundle frequently paired services with 10-15% discount "
            "while maintaining overall margin"
        )
        
        return insights
    
    # Backward compatibility
    @property
    def insights(self) -> list:
        return self.get_insights()
    
    def get_recommendations(self) -> list:
        """Generate actionable recommendations."""
        recommendations = []
        
        if self.rules is not None and len(self.rules) > 0:
            # Top bundling opportunities
            top_3 = self.rules.nlargest(3, 'lift')
            recommendations.append(
                "**Create service packages** for top associations: " +
                ", ".join([f"'{r['rule']}'" for _, r in top_3.iterrows()])
            )
            
            # Technician training
            high_conf_rules = self.rules[self.rules['confidence'] > 0.4].nlargest(5, 'confidence')
            if len(high_conf_rules) > 0:
                recommendations.append(
                    "**Train technicians to suggest**: " +
                    ", ".join([r['consequent'] for _, r in high_conf_rules.iterrows()]) +
                    " when performing related services"
                )
            
            # Installation add-ons
            install_rules = self.rules[
                self.rules['antecedent'].str.contains('Installation', case=False, na=False)
            ].nlargest(3, 'confidence')
            
            if len(install_rules) > 0:
                recommendations.append(
                    "**Installation package**: Always include " +
                    ", ".join([r['consequent'] for _, r in install_rules.iterrows()]) +
                    " as part of installation quotes"
                )
        
        recommendations.append(
            "Create 'Complete Service' packages with 10-15% discount on bundled services - "
            "increases average ticket while providing customer value"
        )
        
        recommendations.append(
            "Update quote templates to automatically suggest complementary services "
            "based on association rules"
        )
        
        recommendations.append(
            "**Next step**: Use Customer Segmentation to identify which segments respond "
            "best to bundled offerings vs à la carte pricing"
        )
        
        return recommendations
    
    # Backward compatibility
    @property
    def recommendations(self) -> list:
        return self.get_recommendations()
    
    @property
    def business_impact(self) -> str:
        return "Service bundling based on association rules can increase average ticket size by 15-25% and improve customer satisfaction by addressing needs proactively"
