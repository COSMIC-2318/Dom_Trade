"""
dashboard.py — DOM Trade Live Dashboard
=========================================
Streamlit dashboard that reads sim_state.json written by
simulate_replay.py and renders a real-time trading dashboard.

Run ALONGSIDE the simulation (two separate terminals):

    Terminal 1:  python simulate_replay.py --input data/new_day.xlsx --speed 0.05
    Terminal 2:  streamlit run dashboard.py

The dashboard auto-refreshes every second to pick up new ticks.
"""

import json
import os
import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

STATE_FILE = "sim_state.json"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DOM Trade — Live",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
      font-family: 'IBM Plex Sans', sans-serif;
      background-color: #0a0e1a;
      color: #e2e8f0;
  }

  /* Header bar */
  .dom-header {
      background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
      border-bottom: 1px solid #1e3a5f;
      padding: 12px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 20px;
      border-radius: 8px;
  }
  .dom-title {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 22px;
      font-weight: 600;
      color: #38bdf8;
      letter-spacing: 2px;
  }
  .dom-subtitle {
      font-size: 12px;
      color: #64748b;
      font-family: 'IBM Plex Mono', monospace;
  }

  /* Metric cards */
  .metric-card {
      background: #0f172a;
      border: 1px solid #1e3a5f;
      border-radius: 10px;
      padding: 16px 20px;
      text-align: center;
  }
  .metric-label {
      font-size: 11px;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      margin-bottom: 6px;
      font-family: 'IBM Plex Mono', monospace;
  }
  .metric-value {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 28px;
      font-weight: 600;
      line-height: 1;
  }
  .metric-sub {
      font-size: 11px;
      color: #94a3b8;
      margin-top: 4px;
      font-family: 'IBM Plex Mono', monospace;
  }

  /* Prediction badge */
  .pred-up {
      background: linear-gradient(135deg, #064e3b, #065f46);
      border: 1px solid #10b981;
      border-radius: 8px;
      padding: 20px;
      text-align: center;
  }
  .pred-down {
      background: linear-gradient(135deg, #4c0519, #881337);
      border: 1px solid #f43f5e;
      border-radius: 8px;
      padding: 20px;
      text-align: center;
  }
  .pred-neutral {
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 20px;
      text-align: center;
  }
  .pred-arrow {
      font-size: 42px;
      line-height: 1;
  }
  .pred-label {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 18px;
      font-weight: 600;
      letter-spacing: 3px;
      margin-top: 6px;
  }
  .pred-score {
      font-size: 12px;
      color: #94a3b8;
      margin-top: 4px;
      font-family: 'IBM Plex Mono', monospace;
  }

  /* Regime badge */
  .regime-bullish  { color: #10b981; }
  .regime-bearish  { color: #f43f5e; }
  .regime-volatile { color: #f59e0b; }
  .regime-other    { color: #94a3b8; }

  /* Progress bar */
  .tick-progress {
      background: #1e293b;
      border-radius: 4px;
      height: 6px;
      margin: 8px 0;
      overflow: hidden;
  }
  .tick-progress-fill {
      height: 100%;
      background: linear-gradient(90deg, #38bdf8, #818cf8);
      border-radius: 4px;
      transition: width 0.3s ease;
  }

  /* Weight bars */
  .weight-row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 6px 0;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 13px;
  }
  .weight-name { width: 70px; color: #94a3b8; }
  .weight-bar-bg {
      flex: 1;
      background: #1e293b;
      border-radius: 3px;
      height: 8px;
      overflow: hidden;
  }
  .weight-bar-fill {
      height: 100%;
      border-radius: 3px;
  }
  .weight-val { width: 40px; text-align: right; color: #e2e8f0; }

  /* Status dot */
  .status-live { color: #10b981; }
  .status-done { color: #f59e0b; }
  .status-wait { color: #64748b; }

  /* Section headers */
  .section-hdr {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 2px;
      color: #475569;
      border-bottom: 1px solid #1e293b;
      padding-bottom: 6px;
      margin-bottom: 12px;
  }

  /* Accuracy pill */
  .acc-good { color: #10b981; }
  .acc-mid  { color: #f59e0b; }
  .acc-bad  { color: #f43f5e; }

  div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; }
  div[data-testid="metric-container"] { background: #0f172a; border: 1px solid #1e3a5f; border-radius: 10px; padding: 12px; }
  .stProgress > div > div { background: linear-gradient(90deg, #38bdf8, #818cf8); }
  footer { visibility: hidden; }
  #MainMenu { visibility: hidden; }
  header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_state():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None


def regime_color(name):
    n = str(name).lower()
    if "bull" in n: return "#10b981"
    if "bear" in n: return "#f43f5e"
    if "vol"  in n: return "#f59e0b"
    return "#94a3b8"


def make_price_chart(price_hist, pred_hist, time_hist):
    fig = make_subplots(rows=1, cols=1)
    n = len(price_hist)
    if n == 0:
        return go.Figure()

    xs = list(range(n))

    # Price line
    fig.add_trace(go.Scatter(
        x=xs, y=price_hist,
        mode="lines",
        line=dict(color="#38bdf8", width=1.5),
        name="Mid Price",
        hovertemplate="₹%{y:.2f}<extra></extra>",
    ))

    # UP/DOWN markers
    up_x   = [i for i, p in enumerate(pred_hist) if p == 1]
    up_y   = [price_hist[i] for i in up_x]
    down_x = [i for i, p in enumerate(pred_hist) if p == 0]
    down_y = [price_hist[i] for i in down_x]

    if up_x:
        fig.add_trace(go.Scatter(
            x=up_x, y=up_y,
            mode="markers",
            marker=dict(symbol="triangle-up", size=6, color="#10b981"),
            name="UP signal",
            hovertemplate="UP ₹%{y:.2f}<extra></extra>",
        ))
    if down_x:
        fig.add_trace(go.Scatter(
            x=down_x, y=down_y,
            mode="markers",
            marker=dict(symbol="triangle-down", size=6, color="#f43f5e"),
            name="DOWN signal",
            hovertemplate="DOWN ₹%{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0f172a",
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        legend=dict(
            orientation="h", x=0, y=1.1,
            font=dict(color="#94a3b8", size=11, family="IBM Plex Mono"),
        ),
        xaxis=dict(
            showgrid=False, showticklabels=False,
            linecolor="#1e293b",
        ),
        yaxis=dict(
            gridcolor="#1e293b", tickfont=dict(color="#64748b", size=10, family="IBM Plex Mono"),
            tickprefix="₹",
        ),
        hovermode="x unified",
    )
    return fig


def make_obi_chart(obi_hist, regime_hist):
    n = len(obi_hist)
    if n == 0:
        return go.Figure()
    xs = list(range(n))
    colors = [regime_color(r) for r in regime_hist]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=xs, y=obi_hist,
        marker_color=colors,
        name="OBI",
        hovertemplate="OBI: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#334155", line_width=1)

    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0f172a",
        margin=dict(l=10, r=10, t=10, b=10),
        height=130,
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, linecolor="#1e293b"),
        yaxis=dict(
            gridcolor="#1e293b",
            tickfont=dict(color="#64748b", size=10, family="IBM Plex Mono"),
            range=[-1, 1],
            zeroline=False,
        ),
    )
    return fig


def make_accuracy_chart(acc_hist):
    n = len(acc_hist)
    if n < 2:
        return go.Figure()
    xs = list(range(n))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=[a * 100 for a in acc_hist],
        mode="lines",
        line=dict(color="#818cf8", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(129,140,248,0.08)",
        name="Accuracy %",
        hovertemplate="%{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="#334155", line_width=1,
                  annotation_text="50%", annotation_font_color="#475569")
    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0f172a",
        margin=dict(l=10, r=10, t=10, b=10),
        height=130,
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(
            gridcolor="#1e293b",
            tickfont=dict(color="#64748b", size=10, family="IBM Plex Mono"),
            ticksuffix="%",
            range=[0, 100],
        ),
    )
    return fig


def make_regime_donut(regime_hist):
    if not regime_hist:
        return go.Figure()
    from collections import Counter
    counts = Counter(regime_hist)
    labels = list(counts.keys())
    values = list(counts.values())
    colors_list = [regime_color(l) for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.6,
        marker=dict(colors=colors_list, line=dict(color="#0a0e1a", width=2)),
        textfont=dict(family="IBM Plex Mono", size=11),
        hovertemplate="%{label}: %{value} ticks (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        margin=dict(l=10, r=10, t=10, b=10),
        height=160,
        showlegend=True,
        legend=dict(
            font=dict(color="#94a3b8", size=11, family="IBM Plex Mono"),
            orientation="v", x=0.75, y=0.5,
        ),
    )
    return fig


# ── Main dashboard ────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="dom-header">
        <div>
            <div class="dom-title">◈ DOM TRADE</div>
            <div class="dom-subtitle">RELIANCE-EQ · NSE · Regime-Aware Forecasting · Day 2 Replay</div>
        </div>
        <div class="dom-subtitle">Angel One SmartAPI · GMM + ARF + FTRL + Hoeffding + PA</div>
    </div>
    """, unsafe_allow_html=True)

    s = load_state()

    # ── No state yet ──────────────────────────────────────────────────────────
    if s is None:
        st.markdown("""
        <div style="text-align:center; padding: 80px 0; color: #475569;">
            <div style="font-size:48px; margin-bottom:16px;">⏳</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:16px; color:#64748b;">
                Waiting for simulation to start...
            </div>
            <div style="font-size:13px; color:#334155; margin-top:12px;">
                Run: <code style="color:#38bdf8;">python simulate_replay.py --input data/new_day.xlsx --speed 0.05</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1)
        st.rerun()
        return

    # ── Status bar ────────────────────────────────────────────────────────────
    tick       = s.get("tick", 0)
    total      = s.get("total_ticks", 1) or 1
    running    = s.get("running", False)
    done       = s.get("done", False)
    pct        = min(tick / total, 1.0)

    status_dot   = "🟢" if running else ("🟡" if done else "⚪")
    status_label = "LIVE REPLAY" if running else ("COMPLETE" if done else "IDLE")

    col_s1, col_s2, col_s3 = st.columns([3, 1, 1])
    with col_s1:
        st.markdown(f"""
        <div style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:#64748b; margin-bottom:4px;">
            {status_dot} {status_label} &nbsp;·&nbsp; Tick {tick:,} / {total:,}
        </div>
        <div class="tick-progress">
            <div class="tick-progress-fill" style="width:{pct*100:.1f}%"></div>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown(f"""
        <div style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:#64748b;">
            Drift Events
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:22px; color:#f59e0b; font-weight:600;">
            {s.get('drift_count', 0)}
        </div>
        """, unsafe_allow_html=True)
    with col_s3:
        pred_total = s.get("total_preds", 0)
        st.markdown(f"""
        <div style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:#64748b;">
            Predictions
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:22px; color:#38bdf8; font-weight:600;">
            {pred_total:,}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)

    # ── Row 1: Key metrics + prediction ──────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns([1.2, 1.2, 1.2, 1.2, 1.8])

    ltp = s.get("ltp", 0)
    mid = s.get("mid_price", 0)
    obi = s.get("obi", 0)
    spread = s.get("spread", 0)
    acc = s.get("accuracy", 0)
    regime = s.get("regime", "—")
    rc = regime_color(regime)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">LTP</div>
            <div class="metric-value" style="color:#38bdf8;">₹{ltp:.2f}</div>
            <div class="metric-sub">Mid ₹{mid:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        obi_color = "#10b981" if obi > 0.05 else ("#f43f5e" if obi < -0.05 else "#f59e0b")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">OBI</div>
            <div class="metric-value" style="color:{obi_color};">{obi:+.3f}</div>
            <div class="metric-sub">Spread ₹{spread:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Regime</div>
            <div class="metric-value" style="color:{rc}; font-size:20px;">{regime}</div>
            <div class="metric-sub">GMM + ARF</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        acc_color = "#10b981" if acc >= 0.75 else ("#f59e0b" if acc >= 0.55 else "#f43f5e")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value" style="color:{acc_color};">{acc:.1%}</div>
            <div class="metric-sub">{s.get('correct',0):,} / {s.get('total_preds',0):,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        pred = s.get("prediction", "—")
        score = s.get("score", 0.5)
        if pred == "UP":
            cls = "pred-up"
            arrow = "▲"
            clr = "#10b981"
        elif pred == "DOWN":
            cls = "pred-down"
            arrow = "▼"
            clr = "#f43f5e"
        else:
            cls = "pred-neutral"
            arrow = "◈"
            clr = "#64748b"
        st.markdown(f"""
        <div class="{cls}">
            <div class="pred-arrow" style="color:{clr};">{arrow}</div>
            <div class="pred-label" style="color:{clr};">{pred}</div>
            <div class="pred-score">score {score:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin: 16px 0 8px 0'></div>", unsafe_allow_html=True)

    # ── Row 2: Price chart ────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">MID PRICE + SIGNALS</div>', unsafe_allow_html=True)
    st.plotly_chart(
        make_price_chart(
            s.get("price_history", []),
            s.get("pred_history", []),
            s.get("time_history", []),
        ),
        use_container_width=True, config={"displayModeBar": False}, key="chart_price"
    )

    # ── Row 3: OBI | Accuracy | Regime donut ─────────────────────────────────
    col_a, col_b, col_c = st.columns([2, 2, 1.5])

    with col_a:
        st.markdown('<div class="section-hdr">ORDER BOOK IMBALANCE (OBI)</div>', unsafe_allow_html=True)
        st.plotly_chart(
            make_obi_chart(s.get("obi_history", []), s.get("regime_history", [])),
            use_container_width=True, config={"displayModeBar": False}, key="chart_obi"
        )

    with col_b:
        st.markdown('<div class="section-hdr">ROLLING ACCURACY</div>', unsafe_allow_html=True)
        st.plotly_chart(
            make_accuracy_chart(s.get("acc_history", [])),
            use_container_width=True, config={"displayModeBar": False}, key="chart_acc"
        )

    with col_c:
        st.markdown('<div class="section-hdr">REGIME DISTRIBUTION</div>', unsafe_allow_html=True)
        st.plotly_chart(
            make_regime_donut(s.get("regime_history", [])),
            use_container_width=True, config={"displayModeBar": False}, key="chart_regime"
        )

    # ── Row 4: Ensemble weights + signal counts ───────────────────────────────
    col_w, col_sig, col_log = st.columns([1.5, 1, 2])

    with col_w:
        st.markdown('<div class="section-hdr">ENSEMBLE WEIGHTS</div>', unsafe_allow_html=True)
        weights = s.get("weights", {"ftrl": 0.33, "hoeff": 0.33, "pa": 0.33})
        model_colors = {"ftrl": "#38bdf8", "hoeff": "#818cf8", "pa": "#f59e0b"}
        model_labels = {"ftrl": "FTRL", "hoeff": "Hoeff", "pa": "PA"}
        for k, v in weights.items():
            bar_w = int(v * 100)
            st.markdown(f"""
            <div class="weight-row">
                <div class="weight-name">{model_labels.get(k,k)}</div>
                <div class="weight-bar-bg">
                    <div class="weight-bar-fill" style="width:{bar_w}%; background:{model_colors.get(k,'#64748b')};"></div>
                </div>
                <div class="weight-val">{v:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_sig:
        st.markdown('<div class="section-hdr">SIGNAL COUNTS</div>', unsafe_allow_html=True)
        up_c   = s.get("up_count", 0)
        dn_c   = s.get("down_count", 0)
        tot_c  = up_c + dn_c or 1
        up_pct = up_c / tot_c * 100
        dn_pct = dn_c / tot_c * 100
        st.markdown(f"""
        <div style="margin-top:8px;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:13px; color:#10b981; margin-bottom:6px;">
                ▲ UP &nbsp; {up_c:,} &nbsp; <span style="color:#64748b;">({up_pct:.0f}%)</span>
            </div>
            <div style="background:#1e293b; border-radius:3px; height:6px; margin-bottom:12px;">
                <div style="width:{up_pct:.0f}%; height:100%; background:#10b981; border-radius:3px;"></div>
            </div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:13px; color:#f43f5e; margin-bottom:6px;">
                ▼ DOWN {dn_c:,} &nbsp; <span style="color:#64748b;">({dn_pct:.0f}%)</span>
            </div>
            <div style="background:#1e293b; border-radius:3px; height:6px;">
                <div style="width:{dn_pct:.0f}%; height:100%; background:#f43f5e; border-radius:3px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_log:
        st.markdown('<div class="section-hdr">LAST TICK LOG</div>', unsafe_allow_html=True)
        ph = s.get("price_history", [])
        oh = s.get("obi_history", [])
        rh = s.get("regime_history", [])
        prh = s.get("pred_history", [])
        th = s.get("time_history", [])
        n_log = min(6, len(ph))
        if n_log > 0:
            rows_html = ""
            for i in range(n_log - 1, max(n_log - 7, -1), -1):
                t_str = str(th[i])[-15:-7] if len(str(th[i])) > 15 else str(th[i])
                p_str = "▲" if prh[i] == 1 else "▼"
                p_color = "#10b981" if prh[i] == 1 else "#f43f5e"
                r_color = regime_color(rh[i])
                rows_html += f"""
                <div style="display:flex; gap:12px; align-items:center;
                            font-family:'IBM Plex Mono',monospace; font-size:11px;
                            padding:3px 0; border-bottom:1px solid #1e293b;">
                    <span style="color:#475569; width:70px;">{t_str}</span>
                    <span style="color:#38bdf8; width:70px;">₹{ph[i]:.2f}</span>
                    <span style="color:#64748b; width:60px;">OBI {oh[i]:+.2f}</span>
                    <span style="color:{r_color}; width:65px;">{rh[i]}</span>
                    <span style="color:{p_color}; width:20px;">{p_str}</span>
                </div>
                """
            st.markdown(rows_html, unsafe_allow_html=True)

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if running:
        time.sleep(1)
        st.rerun()
    elif not done:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()