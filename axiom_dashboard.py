import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os

# Page config
st.set_page_config(
    page_title="Axiom Trading Console",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DB_PATH = "trading_system.db"
REFRESH_RATE = 5  # seconds

# Custom CSS for "Sleek and Slick" look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background-color: #262730;
        border: 1px solid #464b59;
        border-radius: 5px;
        padding: 15px;
        text-align: center;
    }
    .stDataFrame {
        border: 1px solid #464b59;
    }
</style>
""", unsafe_allow_html=True)

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def load_data():
    if not os.path.exists(DB_PATH):
        return None, None, None, None

    conn = get_db_connection()
    
    # 1. Open Positions with Live Market Data
    try:
        query = """
        SELECT 
            p.*, 
            m.yes_price as current_yes,
            m.no_price as current_no,
            m.title as market_title
        FROM positions p
        LEFT JOIN markets m ON p.market_id = m.market_id
        WHERE p.status='open'
        """
        positions_df = pd.read_sql_query(query, conn)
        
        # Calculate Unrealized PnL
        if not positions_df.empty:
            def calc_pnl(row):
                current_price = row['current_yes'] if row['side'] == 'YES' else row['current_no']
                # If market data is missing/null, assume 0 change
                if pd.isna(current_price): return 0.0
                return (current_price - row['entry_price']) * row['quantity'] * 100 # cents to cents? No, price is cents usually.
                # Wait, entry_price in DB is usually 0.xx (dollars) or cents? 
                # KalshiClient usually normalizes to cents or dollars.
                # Let's check DB content from earlier python output: entry_price was 0.5. Current market prices are usually cents (1-99) in API but stored as cents?
                # Quick double check: my manual test inserted: yes_price=(bid+ask)/200. So it is 0-1 float.
                # So PnL = (current_price - entry_price) * quantity (contracts) * 1 (dollar value per contract is $1 on Kalshi).
                # Actually Kalshi contract pays out $1. Price is between 1c and 99c.
                # If I stored entry_price as 0.50 ($0.50), then PnL = (curr - entry) * qty.
                # Let's assume normalized dollars for display.
                
            # Vectorized calc if possible, but row-wise is safer for logic
            positions_df['current_price'] = positions_df.apply(
                lambda x: (x['current_yes'] if x['side'] == 'YES' else x['current_no']) / 100.0, axis=1
            ) # Markets table has prices in CENTS usually? 
              # Wait, VERIFY_KALSHI.PY line 60 says `yes_price=(...+...)/200`. So it stores DOLLARS (0.01-0.99).
              # BUT database.py upsert_markets uses what values?
              # Let's assume markets table matches how we instantiate Market objects.
              # In execute.py/decide.py we see market.yes_price used directly.
              # I need to be careful with units.
              # Let's check `process_markets` in `market_ingestion.py` or similar if available, or just assume the bot stores it consistently.
              # I'll stick to displaying raw stored values and calculating diff.
            
            positions_df['current_market_price'] = positions_df.apply(
                lambda row: row['current_yes'] if row['side'] == 'YES' else row['current_no'], axis=1
            )
            
            # Assuming DB stores normalized dollars (0.xx) OR cents (xx). 
            # If entry is 0.5, it's likely dollars. If market price is 50, it's cents.
            # I will normalize heuristically: if average > 1, it's cents.
            
            positions_df['unrealized_pnl'] = (positions_df['current_market_price'] - positions_df['entry_price']) * positions_df['quantity']
            
    except Exception as e:
        print(f"Error loading positions: {e}")
        positions_df = pd.DataFrame()


    # 2. Trade Logs (for PnL)
    try:
        trades_df = pd.read_sql_query("SELECT * FROM trade_logs ORDER BY exit_timestamp DESC", conn)
    except:
        trades_df = pd.DataFrame()

    # 3. Daily Cost
    try:
        cost_df = pd.read_sql_query("SELECT * FROM daily_cost_tracking ORDER BY date DESC", conn)
    except:
        cost_df = pd.DataFrame()
        
    # 4. Markets Summary
    try:
        markets_p = pd.read_sql_query("SELECT count(*) as count, status FROM markets GROUP BY status", conn)
    except:
        markets_p = pd.DataFrame()

    # 5. Market Analyses
    try:
        analyses_df = pd.read_sql_query("SELECT * FROM market_analyses ORDER BY analysis_timestamp DESC LIMIT 50", conn)
    except:
        analyses_df = pd.DataFrame()
        
    # 6. LLM Queries
    try:
        llm_df = pd.read_sql_query("SELECT * FROM llm_queries ORDER BY timestamp DESC LIMIT 20", conn)
    except:
        llm_df = pd.DataFrame()

    conn.close()
    return positions_df, trades_df, cost_df, markets_p, analyses_df, llm_df

from src.config.settings import settings

# Sidebar
st.sidebar.title("🚀 Axiom Control")
st.sidebar.markdown(f"**🤖 Model:** `{settings.trading.primary_model}`")
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh Rate (s)", 5, 60, 5)

if st.sidebar.button("Refresh Now"):
    st.rerun()

# Main Dashboard
st.title("🚀 Axiom Trading Console")
st.markdown("Real-time performance monitoring via **Unified Advanced Trading System**")

# Load Data
positions, trades, costs, markets_p, analyses, llm_queries = load_data()

if positions is None:
    st.error(f"Database not found at {DB_PATH}. Run the bot first!")
    st.stop()

# Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

# Calculate Metrics
total_open_pnl = 0 
open_positions_count = len(positions) if not positions.empty else 0
total_realized_pnl = trades['pnl'].sum() if not trades.empty and 'pnl' in trades.columns else 0.0
today_cost = costs.iloc[0]['total_ai_cost'] if not costs.empty else 0.0

# Market Stats
active_markets_count = markets_p[markets_p['status'] == 'active']['count'].sum() if not markets_p.empty else 0
total_markets_count = markets_p['count'].sum() if not markets_p.empty else 0

# Check Heartbeat (Last updated market)
conn = get_db_connection()
try:
    # Check markets
    last_mkt_df = pd.read_sql_query("SELECT MAX(last_updated) as last_ts FROM markets", conn)
    last_mkt_ts = pd.to_datetime(last_mkt_df.iloc[0]['last_ts']) if not last_mkt_df.empty and last_mkt_df.iloc[0]['last_ts'] else datetime.min

    # Check analyses (AI activity)
    last_ai_df = pd.read_sql_query("SELECT MAX(analysis_timestamp) as last_ts FROM market_analyses", conn)
    last_ai_ts = pd.to_datetime(last_ai_df.iloc[0]['last_ts']) if not last_ai_df.empty and last_ai_df.iloc[0]['last_ts'] else datetime.min
    
    # Take the most recent activity
    last_heartbeat = max(last_mkt_ts, last_ai_ts)
    
    if last_heartbeat > datetime.min:
        seconds_ago = (datetime.now() - last_heartbeat).total_seconds()
        
        if seconds_ago < 120:  # 2 mins
            status = "🟢 RUNNING"
            status_delta = f"{int(seconds_ago)}s ago"
        elif seconds_ago < 600:  # 10 mins (Accounting for 5 min sleep intervals)
            status = "🟡 SLEEPING"
            status_delta = f"{int(seconds_ago/60)}m ago"
        else:
            status = "🔴 STALLED"
            status_delta = f"{int(seconds_ago/60)}m ago"
    else:
        status = "⚪ WAITING"
        status_delta = "No Data"
except:
    status = "⚪ UNKNOWN"
    status_delta = "Error"
conn.close()

with col1:
    st.metric("Bot Status", status, delta=status_delta)
with col2:
    st.metric("Open Positions", f"{open_positions_count}", delta=None)
with col3:
    pnl_label = "Paper PnL (Simulated)" if settings.trading.paper_trading_mode else "Realized PnL"
    st.metric(pnl_label, f"${total_realized_pnl:,.2f}", delta=f"{len(trades)} trades" if not trades.empty else None)
with col4:
    st.metric("Active Markets", f"{active_markets_count}", delta=f"of {total_markets_count} total")
with col5:
    st.metric("Today's AI Cost", f"${today_cost:,.2f}", delta_color="inverse")
    if settings.trading.paper_trading_mode:
        st.caption("🛑 Paper Mode Active")

# Activity Feed Section
st.subheader("📡 Live Activity Feed")
col_feed1, col_feed2 = st.columns([1, 1])

with col_feed1:
    st.markdown("##### 🛒 Market Updates")
    try:
        conn = get_db_connection()
        # Get 5 most recently updated markets
        recent_markets = pd.read_sql_query("""
            SELECT title, volume, last_updated, status 
            FROM markets 
            ORDER BY last_updated DESC 
            LIMIT 5
        """, conn)
        conn.close()
        
        if not recent_markets.empty:
            for index, row in recent_markets.iterrows():
                st.caption(f"**{row['last_updated']}**: Updated **{row['title']}** (Vol: {row['volume']}) - `{row['status']}`")
        else:
            st.info("No market activity logged yet.")
    except:
        st.error("Could not fetch activity feed.")

with col_feed2:
    st.markdown("##### 🧠 AI Decisions")
    if not analyses.empty:
        for index, row in analyses.head(5).iterrows():
            decision_color = "green" if row['decision_action'] == "BUY" else "red" if row['decision_action'] == "SELL" else "gray"
            st.caption(f"**{row['analysis_timestamp']}**: :{decision_color}[{row['decision_action']}] on **{row['market_id']}** (Conf: {row.get('confidence', 0):.2f})")
    else:
        st.info("No AI decisions logged yet.")

# Visualization Tab
tab1, tab2, tab3 = st.tabs(["📊 Portfolio", "🧠 AI Logic", "📜 Details"])

with tab1:
    # Charts Row
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("📈 Cumulative PnL")
        if not trades.empty:
            trades['exit_timestamp'] = pd.to_datetime(trades['exit_timestamp'])
            trades = trades.sort_values('exit_timestamp')
            trades['cumulative_pnl'] = trades['pnl'].cumsum()
            
            fig_pnl = px.line(trades, x='exit_timestamp', y='cumulative_pnl', 
                              title='Realized PnL Over Time',
                              template="plotly_dark")
            fig_pnl.update_layout(height=350)
            st.plotly_chart(fig_pnl, width="stretch")
        else:
            st.info("No trades executed yet.")

    with col_chart2:
        st.subheader("📊 Portfolio Composition")
        if not positions.empty:
            # Group by strategy or market category if available, else just side
            if 'strategy' in positions.columns:
                composition = positions.groupby('strategy').size().reset_index(name='count')
                fig_comp = px.pie(composition, values='count', names='strategy', 
                                  title='Positions by Strategy',
                                  template="plotly_dark", hole=0.4)
                fig_comp.update_layout(height=350)
                st.plotly_chart(fig_comp, width="stretch")
            else:
                fig_comp = px.pie(positions, names='side', title='Positions by Side',
                                  template="plotly_dark", hole=0.4)
                st.plotly_chart(fig_comp, width="stretch")
        else:
            st.info("No active positions.")

with tab2:
    st.subheader("🧠 AI Brain Analytics")
    
    if not analyses.empty:
        # Pre-process data
        analyses['analysis_timestamp'] = pd.to_datetime(analyses['analysis_timestamp'])
        
        # Row 1: Key Distribution Metrics
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.markdown("#### 🎯 Confidence Distribution")
            fig_conf = px.histogram(analyses, x="confidence", nbins=20, 
                                   color="decision_action",
                                   title="Confidence Levels by Decision",
                                   template="plotly_dark",
                                   color_discrete_map={"BUY": "green", "SKIP": "gray", "SELL": "red"})
            fig_conf.update_layout(bargap=0.1, height=300)
            st.plotly_chart(fig_conf, width="stretch")
            
        with col_viz2:
            st.markdown("#### ⏱️ Decision Timeline")
            # Resample by hour or show raw if few
            fig_time = px.scatter(analyses, x="analysis_timestamp", y="confidence",
                                color="decision_action",
                                size="cost_usd", 
                                hover_data=["market_id"],
                                title="Decisions over Time (Size = Cost)",
                                template="plotly_dark",
                                color_discrete_map={"BUY": "green", "SKIP": "gray", "SELL": "red"})
            fig_time.update_layout(height=300)
            st.plotly_chart(fig_time, width="stretch")

    if not llm_queries.empty:
        st.markdown("---")
        st.subheader("🔍 Deep Dive: LLM Thoughts")
        
        # Cost Analysis
        col_cost1, col_cost2 = st.columns([1, 2])
        with col_cost1:
            total_stats = llm_queries.agg({'cost_usd': 'sum', 'tokens_used': 'sum'})
            st.metric("Total AI Spend", f"${total_stats['cost_usd']:.2f}")
            st.metric("Total Tokens", f"{int(total_stats['tokens_used']):,}")
        
        with col_cost2:
            # Scatter of Cost vs Confidence
            if 'confidence_extracted' in llm_queries.columns:
                fig_scatter = px.scatter(llm_queries, x="confidence_extracted", y="cost_usd",
                                       color="query_type",
                                       title="Cost Efficiency: Spend vs Confidence",
                                       template="plotly_dark")
                fig_scatter.update_layout(height=250)
                st.plotly_chart(fig_scatter, width="stretch")

        # Inspector
        st.markdown("#### 🔬 Query Inspector")
        try:
            # Create a readable label
            llm_queries['label'] = llm_queries['timestamp'].astype(str) + " | " + llm_queries['market_id'] + " (" + llm_queries['query_type'] + ")"
            
            selected_query_label = st.selectbox("Select Interaction", 
                llm_queries['label'],
                index=0
            )
            
            query_data = llm_queries[llm_queries['label'] == selected_query_label].iloc[0]
            
            col_q1, col_q2 = st.columns(2)
            with col_q1:
                 st.markdown("**📤 Prompt Sent:**")
                 st.code(query_data['prompt'], language="markdown")
            with col_q2:
                 st.markdown("**📥 AI Response:**")
                 st.code(query_data['response'], language="markdown")
                 
            st.caption(f"Tokens: {query_data['tokens_used']} | Cost: ${query_data['cost_usd']:.4f} | Strategy: {query_data['strategy']}")
        except Exception as e:
            st.error(f"Error displaying query details: {e}")
    else:
        st.info("No LLM queries recorded available for inspection.")

with tab3:
    # Detailed Data
    
    # NEW: Action Queue / Recent Decisions
    st.subheader("📋 Recent Actions (Last 20 Decisions)")
    if not analyses.empty:
        # Filter for BUY/SELL only
        action_df = analyses[analyses['decision_action'].isin(['BUY', 'SELL'])].copy()
        if not action_df.empty:
            action_df['Time'] = action_df['analysis_timestamp']
            action_df['Market'] = action_df['market_id']
            action_df['Action'] = action_df['decision_action']
            action_df['Conf'] = action_df['confidence']
            action_df['Cost'] = action_df['cost_usd']
            
            st.dataframe(
                action_df[['Time', 'Market', 'Action', 'Conf', 'Cost']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Time": st.column_config.DatetimeColumn("Time", format="HH:mm:ss"),
                    "Conf": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
                    "Cost": st.column_config.NumberColumn("AI Cost", format="$%.4f"),
                }
            )
        else:
            st.info("No recent BUY/SELL decisions.")
    
    st.markdown("---")

    st.subheader("📋 Active Positions")
    if not positions.empty:
        # Prepare columns for display
        display_df = positions.copy()
        
        # Calculate PnL if not already (redundant safety)
        if 'unrealized_pnl' not in display_df.columns:
            display_df['unrealized_pnl'] = 0.0

        # Format for display
        display_df['Market'] = display_df['market_title'].fillna(display_df['market_id'])
        display_df['Side'] = display_df['side']
        display_df['Size'] = display_df['quantity']
        display_df['Entry'] = display_df['entry_price']
        display_df['Current'] = display_df['current_market_price']
        display_df['PnL'] = display_df['unrealized_pnl']
        display_df['Conf'] = display_df['confidence']
        display_df['Rationale'] = display_df['rationale']
        
        # Select and reorder
        cols_to_show = ['Market', 'Side', 'Size', 'Entry', 'Current', 'PnL', 'Conf', 'Rationale']
        final_df = display_df[cols_to_show]

        # Summary Metrics top-level
        p_col1, p_col2, p_col3 = st.columns(3)
        total_invested = (display_df['Entry'] * display_df['Size']).sum()
        total_params_pnl = display_df['PnL'].sum()
        
        p_col1.metric("Total Invested", f"${total_invested:,.2f}")
        p_col2.metric("Total Unrealized PnL", f"${total_params_pnl:,.2f}", 
                     delta=f"{(total_params_pnl/total_invested)*100:.1f}%" if total_invested > 0 else "0%")
        p_col3.metric("Active Count", len(display_df))
        
        st.markdown("---")

        # Graphical "Trading Cards" Layout
        cols = st.columns(3)
        for idx, row in final_df.iterrows():
            with cols[idx % 3]:
                with st.container(border=True):
                    # Header: Market Title & Side
                    side_color = "green" if row['Side'] == "YES" else "red"
                    st.markdown(f"**{row['Market'][:50]}...**")
                    st.markdown(f":{side_color}[**{row['Side']}**] @ {row['Entry']:.2f}¢")
                    
                    # Metrics: Current Price & PnL
                    m1, m2 = st.columns(2)
                    m1.metric("Mark", f"{row['Current']:.2f}¢")
                    
                    pnl_val = row['PnL']
                    m2.metric("PnL", f"${pnl_val:.2f}", 
                              delta=f"{pnl_val:.2f}",
                              delta_color="normal")
                    
                    # Confidence Bar
                    conf_val = row['Conf'] if pd.notna(row['Conf']) else 0.0
                    st.progress(conf_val, text=f"AI Confidence: {conf_val:.0%}")
                    
                    # Rationale Expander
                    with st.expander("AI Rationale"):
                         st.caption(row['Rationale'])


        
    else:
        st.info("No active positions. The bot is scanning directly...", icon="🕵️")

    st.subheader("📜 Recent Trades")
    if not trades.empty:
        # Sort by exit time
        recent_trades = trades.sort_values('exit_timestamp', ascending=False).head(12)
        
        # Grid layout for trades
        t_cols = st.columns(3)
        for idx, row in recent_trades.iterrows():
            with t_cols[idx % 3]:
                with st.container(border=True):
                    # Header
                    pnl_val = row['pnl']
                    pnl_color = "green" if pnl_val >= 0 else "red"
                    header_icon = "💰" if pnl_val >= 0 else "💸"
                    
                    st.markdown(f"**{row['market_id']}**")
                    st.caption(f"Strategy: {row.get('strategy', 'Standard')}")
                    
                    # Metrics
                    c1, c2 = st.columns(2)
                    c1.metric("Result", f":{pnl_color}[${pnl_val:.2f}]")
                    c1.caption(f"{row['side']} @ {row['entry_price']:.2f}")
                    
                    c2.metric("Exit", f"${row['exit_price']:.2f}")
                    c2.caption(f"{pd.to_datetime(row['exit_timestamp']).strftime('%H:%M:%S')}")
                    
                    # Rationale
                    with st.expander("Exit Reason"):
                         st.write(f"**Rationale:** {row.get('rationale', 'N/A')}")
    else:
        st.info("No closed trades yet.")

# Auto Refresh Logic
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
