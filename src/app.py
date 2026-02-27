# app.py
# Streamlit application for interactive VarProIRLS inspection.
#
# Run:
#   streamlit run app.py
#
# All state lives in st.session_state so Streamlit reruns never
# lose computed results.

import io
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st

from solver import VarProIRLS

# ======================================================================
# Page config — must be first Streamlit call
# ======================================================================
st.set_page_config(
    page_title="VarProIRLS Inspector",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# Minimal CSS: monospace values, tighter spacing, dark grid
# ======================================================================
st.markdown("""
<style>
  .metric-box  { font-family: monospace; font-size: 0.85rem; }
  .stProgress > div > div { background-color: #4a9eff; }
  div[data-testid="stSidebar"] { min-width: 320px; }
</style>
""", unsafe_allow_html=True)


# ======================================================================
# Colour palette — one colour per model, consistent across all plots
# ======================================================================
MODEL_COLORS = {
    'exponential':      '#4a9eff',
    'sqrt_exponential': '#ff8c42',
    'power_law':        '#50c878',
}
MODEL_LABELS = {
    'exponential':      'Exp(e^{-Bx})',
    'sqrt_exponential': 'SqrtExp(e^{-B√x})',
    'power_law':        'Power(x^{-B})',
}


# ======================================================================
# Helpers
# ======================================================================

def safe_read_csv(uploaded_file):
    """Read uploaded CSV; return (DataFrame, error_string)."""
    try:
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as exc:
        return None, str(exc)


def validate_columns(df, x_col, y_col, label):
    """Return error string or None."""
    for col in [x_col, y_col]:
        if col not in df.columns:
            return f"[{label}] Column '{col}' not found. Available: {list(df.columns)}"
        try:
            pd.to_numeric(df[col])
        except Exception:
            return f"[{label}] Column '{col}' cannot be converted to numeric."
    if df[[x_col, y_col]].isnull().any().any():
        return f"[{label}] Columns contain NaN values."
    return None


def make_iteration_fig(fitter, model, iter_idx):
    """
    Build the main 4-panel figure for a given model at a given iteration.

    Layout:
      [0,0] Fit curve + data + asymptote + uncertainty band
      [0,1] Parameter traces (B, C_unscaled, A) vs iteration
      [1,0] Weight bar chart
      [1,1] Objective traces (weighted_ssr, rel_obj, rel_w) vs iteration
    """
    hist    = fitter.iteration_history[model]
    res     = fitter.results[model]
    params  = hist['params']
    weights_hist = hist['weights']
    obj_hist     = hist['objective']

    n_iters = len(params)
    idx     = min(iter_idx, n_iters - 1)

    # ---- Unpack history up to idx ----
    iters_so_far = [p['iter']       for p in params[:idx + 1]]
    B_trace      = [p['B']          for p in params[:idx + 1]]
    C_trace      = [p['C_unscaled'] for p in params[:idx + 1]]
    A_trace      = [p['A']          for p in params[:idx + 1]]
    ssr_trace    = [o['weighted_ssr'] for o in obj_hist[:idx + 1]]
    relobj_trace = [o['rel_obj']      for o in obj_hist[:idx + 1]]
    relw_trace   = [o['rel_w']        for o in obj_hist[:idx + 1]]

    cur_B = params[idx]['B']
    cur_C = params[idx]['C']
    cur_A = params[idx]['A']
    cur_C_u = params[idx]['C_unscaled']
    cur_weights = weights_hist[idx]
    color = MODEL_COLORS[model]

    # ---- Compute current model curve ----
    x_plot = np.linspace(fitter.x_min, fitter.x_max * 1.5, 300)
    t_plot = x_plot / fitter.x_max
    fitter.model_type = model
    phi_plot = fitter._compute_basis(cur_B, t_plot)
    y_plot_scaled = cur_C + cur_A * phi_plot[:, 1]
    y_plot = fitter.unscale_y(y_plot_scaled)

    x_data = fitter.raw_x
    y_data = fitter.unscale_y(fitter.raw_y)

    sigma_u = fitter.y_range * res.get('sigma_mc', 0.0)

    # ---- Build figure ----
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"Fit — {MODEL_LABELS[model]}  (iter {idx})",
            "Parameter traces",
            f"IRLS weights  (iter {idx})",
            "Convergence objectives",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    # ── Panel [1,1]: Fit curve ──────────────────────────────────────────
    # Data points
    fig.add_trace(go.Scatter(
        x=x_data, y=y_data,
        mode='markers',
        marker=dict(color='white', size=8, line=dict(color='gray', width=1)),
        name='Data',
        showlegend=True,
    ), row=1, col=1)

    # Model curve
    fig.add_trace(go.Scatter(
        x=x_plot, y=y_plot,
        mode='lines',
        line=dict(color=color, width=2),
        name='Model',
        showlegend=True,
    ), row=1, col=1)

    # Asymptote line
    fig.add_trace(go.Scatter(
        x=[fitter.x_min, fitter.x_max * 1.5],
        y=[cur_C_u, cur_C_u],
        mode='lines',
        line=dict(color=color, width=1.5, dash='dash'),
        name=f'C = {cur_C_u:.6f}',
        showlegend=True,
    ), row=1, col=1)

    # Uncertainty band (only if UQ is available)
    if sigma_u > 0:
        fig.add_trace(go.Scatter(
            x=[fitter.x_min, fitter.x_max * 1.5,
               fitter.x_max * 1.5, fitter.x_min],
            y=[cur_C_u + sigma_u, cur_C_u + sigma_u,
               cur_C_u - sigma_u, cur_C_u - sigma_u],
            fill='toself',
            fillcolor=f'rgba({_hex_to_rgb(color)},0.12)',
            line=dict(width=0),
            name=f'±{sigma_u:.2e}',
            showlegend=True,
        ), row=1, col=1)

    # Reference truth (if provided via err_df)
    truth = st.session_state.get('truth_val')
    if truth is not None:
        fig.add_trace(go.Scatter(
            x=[fitter.x_min, fitter.x_max * 1.5],
            y=[truth, truth],
            mode='lines',
            line=dict(color='red', width=1.5, dash='dot'),
            name=f'Ref = {truth:.6f}',
            showlegend=True,
        ), row=1, col=1)

    # ── Panel [1,2]: Parameter traces ──────────────────────────────────
    # B on primary y-axis
    fig.add_trace(go.Scatter(
        x=iters_so_far, y=B_trace,
        mode='lines+markers', marker=dict(size=5),
        line=dict(color='#4a9eff'), name='B',
    ), row=1, col=2)

    # C_unscaled on secondary y-axis (add as second y-axis manually)
    fig.add_trace(go.Scatter(
        x=iters_so_far, y=C_trace,
        mode='lines+markers', marker=dict(size=5),
        line=dict(color='#ff6b6b'), name='C (unscaled)',
        yaxis='y3',
    ), row=1, col=2)

    # A on same axis as B (usually small)
    fig.add_trace(go.Scatter(
        x=iters_so_far, y=A_trace,
        mode='lines+markers', marker=dict(size=5),
        line=dict(color='#50c878', dash='dot'), name='A',
    ), row=1, col=2)

    # ── Panel [2,1]: Weight bar chart ───────────────────────────────────
    n = len(cur_weights)
    fig.add_trace(go.Bar(
        x=list(range(n)),
        y=cur_weights,
        marker_color=color,
        name='Weights',
        showlegend=False,
    ), row=2, col=1)

    # Vertical labels for x-axis: actual x values
    tick_labels = [f"{v:.1f}" for v in fitter.raw_x]
    fig.update_xaxes(
        tickvals=list(range(n)),
        ticktext=tick_labels,
        title_text="Basis size",
        row=2, col=1,
    )
    fig.update_yaxes(title_text="Weight", row=2, col=1)

    # ── Panel [2,2]: Convergence objectives ────────────────────────────
    if ssr_trace:
        fig.add_trace(go.Scatter(
            x=iters_so_far, y=ssr_trace,
            mode='lines+markers', marker=dict(size=5),
            line=dict(color='#ffd700'), name='Weighted SSR',
        ), row=2, col=2)

    if len(relobj_trace) > 1:
        fig.add_trace(go.Scatter(
            x=iters_so_far[1:], y=relobj_trace[1:],
            mode='lines+markers', marker=dict(size=5),
            line=dict(color='#ff8c42', dash='dash'), name='rel_obj',
        ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=iters_so_far[1:], y=relw_trace[1:],
            mode='lines+markers', marker=dict(size=5),
            line=dict(color='#c084fc', dash='dot'), name='rel_w',
        ), row=2, col=2)
        fig.update_yaxes(type='log', row=2, col=2)

    # ── Global layout ───────────────────────────────────────────────────
    fig.update_layout(
        height=760,
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        legend=dict(
            bgcolor='rgba(30,30,40,0.8)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(t=60, b=40, l=50, r=30),
    )

    return fig


def _hex_to_rgb(hex_color):
    """'#4a9eff' -> '74,158,255'  (for rgba strings)."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


def make_final_summary_fig(fitter, models):
    """
    Single figure showing all fitted curves + asymptotes + uncertainty
    bands overlaid on the data. Used in the Results tab.
    """
    fig = go.Figure()

    x_data = fitter.raw_x
    y_data = fitter.unscale_y(fitter.raw_y)

    fig.add_trace(go.Scatter(
        x=x_data, y=y_data,
        mode='markers',
        marker=dict(color='white', size=9, line=dict(color='gray', width=1)),
        name='Data',
    ))

    x_plot = np.linspace(fitter.x_min, fitter.x_max * 1.6, 400)
    t_plot = x_plot / fitter.x_max

    for model in models:
        if model not in fitter.results:
            continue
        res   = fitter.results[model]
        color = MODEL_COLORS[model]
        label = MODEL_LABELS[model]

        fitter.model_type = model
        phi  = fitter._compute_basis(res['B'], t_plot)
        y_sc = res['C'] + res['A'] * phi[:, 1]
        y_pl = fitter.unscale_y(y_sc)
        C_u  = fitter.unscale_y(res['C'])
        sig  = fitter.y_range * res.get('sigma_mc', 0.0)

        fig.add_trace(go.Scatter(
            x=x_plot, y=y_pl,
            mode='lines',
            line=dict(color=color, width=2.5),
            name=label,
        ))
        fig.add_hline(
            y=C_u,
            line=dict(color=color, width=1, dash='dash'),
            annotation_text=f"C={C_u:.7f}",
            annotation_font=dict(color=color, size=11),
        )
        if sig > 0:
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_plot, x_plot[::-1]]),
                y=np.concatenate([
                    np.full(len(x_plot), C_u + sig),
                    np.full(len(x_plot), C_u - sig),
                ]),
                fill='toself',
                fillcolor=f'rgba({_hex_to_rgb(color)},0.10)',
                line=dict(width=0),
                name=f'{label} ±σ',
                showlegend=False,
            ))

    truth = st.session_state.get('truth_val')
    if truth is not None:
        fig.add_hline(
            y=truth,
            line=dict(color='red', width=1.5, dash='dot'),
            annotation_text=f"Ref={truth:.7f}",
            annotation_font=dict(color='red', size=11),
        )

    fig.update_layout(
        height=550,
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        xaxis_title="Basis size",
        yaxis_title=fitter.y_col,
        title="All Models — Final Fit",
        legend=dict(bgcolor='rgba(30,30,40,0.8)', bordercolor='gray', borderwidth=1),
        margin=dict(t=60, b=40, l=60, r=30),
    )
    return fig


# ======================================================================
# Sidebar — data upload + configuration
# ======================================================================

def sidebar():
    st.sidebar.title("⚙  Configuration")

    # ------------------------------------------------------------------ #
    # 1. Data upload
    # ------------------------------------------------------------------ #
    st.sidebar.header("1 · Data")

    init_file = st.sidebar.file_uploader("init.csv  (required)", type="csv", key="init_upload")
    inf_file  = st.sidebar.file_uploader("inf.csv   (optional)", type="csv", key="inf_upload")
    err_file  = st.sidebar.file_uploader("err.csv   (optional)", type="csv", key="err_upload")

    cfg = {}

    # Parse init.csv
    if init_file:
        df_init, err = safe_read_csv(init_file)
        if err:
            st.sidebar.error(f"init.csv: {err}")
            cfg['data_ready'] = False
            return cfg
        cfg['df_init'] = df_init
        cols = list(df_init.columns)

        st.sidebar.markdown("**Column mapping — init.csv**")
        cfg['x_col'] = st.sidebar.selectbox("x  (basis size)", cols,
                                             index=0, key="x_col")
        cfg['y_col'] = st.sidebar.selectbox("y  (expectation value)", cols,
                                             index=min(1, len(cols) - 1), key="y_col")

        # Validate
        verr = validate_columns(df_init, cfg['x_col'], cfg['y_col'], "init.csv")
        if verr:
            st.sidebar.error(verr)
            cfg['data_ready'] = False
            return cfg
    else:
        st.sidebar.info("Upload init.csv to begin.")
        cfg['data_ready'] = False
        return cfg

    # Parse inf.csv (optional — provides truth value for reference line)
    cfg['df_inf'] = None
    cfg['truth_val'] = None
    if inf_file:
        df_inf, err = safe_read_csv(inf_file)
        if err:
            st.sidebar.warning(f"inf.csv parse error: {err}")
        else:
            cfg['df_inf'] = df_inf
            inf_cols = list(df_inf.columns)
            if cfg['y_col'] in inf_cols:
                try:
                    cfg['truth_val'] = float(df_inf[cfg['y_col']].iloc[-1])
                    st.sidebar.markdown(
                        f"**Reference value** ({cfg['y_col']}): `{cfg['truth_val']:.8f}`"
                    )
                except Exception:
                    pass

    # Parse err.csv (optional — provides reference error band)
    cfg['df_err'] = None
    if err_file:
        df_err, err = safe_read_csv(err_file)
        if err:
            st.sidebar.warning(f"err.csv parse error: {err}")
        else:
            cfg['df_err'] = df_err

    # ------------------------------------------------------------------ #
    # 2. Solver configuration
    # ------------------------------------------------------------------ #
    st.sidebar.header("2 · Solver")

    cfg['models'] = st.sidebar.multiselect(
        "Models to fit",
        ['exponential', 'sqrt_exponential', 'power_law'],
        default=['exponential', 'sqrt_exponential', 'power_law'],
        key="models_sel",
    )
    if not cfg['models']:
        st.sidebar.warning("Select at least one model.")
        cfg['data_ready'] = False
        return cfg

    cfg['max_iter']   = st.sidebar.slider("Max iterations",   10, 300, 110, 10)
    cfg['damping']    = st.sidebar.slider("Damping factor",   0.05, 1.0, 0.5, 0.05)
    cfg['tol']        = float(st.sidebar.select_slider(
        "Convergence tolerance",
        options=[1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11],
        value=1e-9,
    ))
    cfg['n_bootstrap']       = st.sidebar.slider("Bootstrap samples", 10, 200, 40, 10)
    cfg['confidence_level']  = st.sidebar.slider("Confidence level (%)", 80, 99, 95, 1)

    cfg['data_ready'] = True
    return cfg


# ======================================================================
# Main content
# ======================================================================

def main():
    st.title("VarProIRLS — Interactive Convergence Inspector")

    cfg = sidebar()

    if not cfg.get('data_ready'):
        st.markdown("""
        ### Getting started
        1. Upload **init.csv** in the sidebar (columns: basis size, expectation value).
        2. Optionally upload **inf.csv** (reference/truth value) and **err.csv** (error bars).
        3. Configure the solver settings.
        4. Click **Run Fitting**.
        """)
        return

    # Persist truth_val in session so make_iteration_fig can access it
    st.session_state['truth_val'] = cfg.get('truth_val')

    # ------------------------------------------------------------------ #
    # Run button
    # ------------------------------------------------------------------ #
    col_run, col_clear = st.columns([2, 1])
    with col_run:
        run_btn = st.button("▶  Run Fitting", type="primary", use_container_width=True)
    with col_clear:
        if st.button("✕  Clear Results", use_container_width=True):
            for key in ['fitter', 'fit_done', 'selected_model']:
                st.session_state.pop(key, None)
            st.rerun()

    if run_btn:
        st.session_state.pop('fitter', None)
        st.session_state['fit_done'] = False

        progress_bar  = st.progress(0, text="Initialising…")
        status_text   = st.empty()
        total_models  = len(cfg['models'])

        def progress_cb(model, k, max_iter):
            model_idx = cfg['models'].index(model)
            frac = (model_idx + (k + 1) / max_iter) / total_models
            progress_bar.progress(min(frac, 1.0),
                                  text=f"Fitting {model}  —  iter {k + 1}")

        with st.spinner("Running IRLS…"):
            try:
                fitter = VarProIRLS(
                    df=cfg['df_init'],
                    x_col=cfg['x_col'],
                    y_col=cfg['y_col'],
                    err_df=cfg.get('df_err'),
                )
                fitter.fit_irls(
                    models=cfg['models'],
                    max_iter=cfg['max_iter'],
                    damping=cfg['damping'],
                    tol=cfg['tol'],
                    compute_uq=True,
                    progress_callback=progress_cb,
                )
                # Re-run uncertainty with UI-configured params
                fitter.compute_uncertainty(
                    n_bootstrap=cfg['n_bootstrap'],
                    confidence_level=cfg['confidence_level'],
                )

                st.session_state['fitter']  = fitter
                st.session_state['fit_done'] = True

            except Exception as exc:
                st.error(f"Fitting failed: {exc}")
                import traceback
                st.code(traceback.format_exc())

        progress_bar.empty()
        status_text.empty()
        if st.session_state.get('fit_done'):
            st.success("Fitting complete.")

    # ------------------------------------------------------------------ #
    # Show results (always if fitter exists in session)
    # ------------------------------------------------------------------ #
    if not st.session_state.get('fit_done') or 'fitter' not in st.session_state:
        return

    fitter  = st.session_state['fitter']
    models  = list(fitter.results.keys())

    tab_names = ["📊 Summary", "🔬 Iteration Inspector", "📈 Final Curves", "📋 Data Table"]
    tabs = st.tabs(tab_names)

    # ================================================================== #
    # TAB 0 — Summary metrics
    # ================================================================== #
    with tabs[0]:
        st.subheader("Fitted Parameters")

        header_cols = st.columns([2, 1.5, 1.5, 1.5, 2, 2, 2])
        headers = ["Model", "B", "C (scaled)", "A", "C (original)", "σ_mc", "SSR"]
        for col, h in zip(header_cols, headers):
            col.markdown(f"**{h}**")

        for model in models:
            res   = fitter.results[model]
            C_u   = fitter.unscale_y(res['C'])
            sigma = fitter.y_range * res.get('sigma_mc', 0.0)
            cols  = st.columns([2, 1.5, 1.5, 1.5, 2, 2, 2])
            vals  = [
                f"`{model}`",
                f"`{res['B']:.6f}`",
                f"`{res['C']:.8f}`",
                f"`{res['A']:.6f}`",
                f"`{C_u:.8f}`",
                f"`{sigma:.3e}`",
                f"`{res['ssr']:.3e}`",
            ]
            for col, v in zip(cols, vals):
                col.markdown(v)

        truth = st.session_state.get('truth_val')
        if truth is not None:
            st.markdown("---")
            st.subheader("Distance from Reference")
            ref_cols = st.columns(len(models))
            for col, model in zip(ref_cols, models):
                C_u = fitter.unscale_y(fitter.results[model]['C'])
                col.metric(
                    label=model,
                    value=f"{C_u:.8f}",
                    delta=f"Δ = {C_u - truth:+.3e}",
                )

        st.markdown("---")
        st.subheader("Convergence Summary")
        for model in models:
            hist   = fitter.iteration_history[model]
            n_iter = len(hist['params'])
            final  = hist['objective'][-1] if hist['objective'] else {}
            with st.expander(f"{model}  —  {n_iter} iterations"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Total iterations", n_iter)
                c2.metric("Final rel_obj", f"{final.get('rel_obj', 0):.3e}")
                c3.metric("Final rel_w",   f"{final.get('rel_w',   0):.3e}")

    # ================================================================== #
    # TAB 1 — Iteration inspector
    # ================================================================== #
    with tabs[1]:
        st.subheader("Iteration-by-Iteration Inspection")

        # Model picker
        sel_model = st.selectbox(
            "Model",
            models,
            index=0,
            key="sel_model_iter",
        )
        st.session_state['selected_model'] = sel_model

        hist    = fitter.iteration_history[sel_model]
        n_iters = len(hist['params'])

        if n_iters == 0:
            st.warning("No iteration history available for this model.")
        else:
            # ---- Controls ----
            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([3, 1, 1])
            with ctrl_col1:
                iter_idx = st.slider(
                    "Iteration",
                    min_value=0,
                    max_value=n_iters - 1,
                    value=n_iters - 1,
                    key="iter_slider",
                )
            with ctrl_col2:
                play_speed = st.selectbox("Speed", [0.05, 0.1, 0.2, 0.5, 1.0],
                                          index=2, key="play_speed",
                                          format_func=lambda x: f"{x}s/frame")
            with ctrl_col3:
                play_btn = st.button("▶  Play", key="play_btn")

            # Inline parameter readout
            p = hist['params'][iter_idx]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("B",            f"{p['B']:.8f}")
            m2.metric("C (scaled)",   f"{p['C']:.8f}")
            m3.metric("C (original)", f"{p['C_unscaled']:.8f}")
            m4.metric("A",            f"{p['A']:.6f}")

            # ---- Static plot at selected iteration ----
            # (Plotly figure is always rendered; play just changes slider)
            plot_placeholder = st.empty()

            def render_at(i):
                fig = make_iteration_fig(fitter, sel_model, i)
                plot_placeholder.plotly_chart(fig, use_container_width=True,
                                              config={'displayModeBar': True})

            render_at(iter_idx)

            # ---- Play animation (pure Python loop) ----
            if play_btn:
                for i in range(iter_idx, n_iters):
                    render_at(i)
                    time.sleep(play_speed)

    # ================================================================== #
    # TAB 2 — Final curves
    # ================================================================== #
    with tabs[2]:
        st.subheader("Final Fitted Curves — All Models")
        fig_final = make_final_summary_fig(fitter, models)
        st.plotly_chart(fig_final, use_container_width=True)

        # Zoomed tail view
        st.subheader("Zoomed Tail & Extrapolation")
        zoom_fig = go.Figure(fig_final)
        zoom_start = fitter.x_min + 0.6 * fitter.range_x
        zoom_fig.update_xaxes(range=[zoom_start, fitter.x_max * 1.5])

        # Compute y-range in zoom window
        x_data = fitter.raw_x
        y_data = fitter.unscale_y(fitter.raw_y)
        mask   = x_data >= zoom_start
        if np.any(mask):
            y_lo = float(np.min(y_data[mask]))
            y_hi = float(np.max(y_data[mask]))
            # Include asymptotes in y range
            for model in models:
                res  = fitter.results[model]
                C_u  = float(fitter.unscale_y(res['C']))
                sig  = float(fitter.y_range * res.get('sigma_mc', 0.0))
                y_lo = min(y_lo, C_u - sig * 2)
                y_hi = max(y_hi, C_u + sig * 2)
            span = y_hi - y_lo or 0.01
            zoom_fig.update_yaxes(range=[y_lo - 0.1 * span, y_hi + 0.1 * span])

        zoom_fig.update_layout(title="Zoomed Tail & Extrapolation", height=450)
        st.plotly_chart(zoom_fig, use_container_width=True)

    # ================================================================== #
    # TAB 3 — Data tables
    # ================================================================== #
    with tabs[3]:
        st.subheader("Input Data")
        st.dataframe(cfg['df_init'], use_container_width=True)

        if cfg.get('df_inf') is not None:
            st.subheader("Reference (inf.csv)")
            st.dataframe(cfg['df_inf'], use_container_width=True)

        if cfg.get('df_err') is not None:
            st.subheader("Error (err.csv)")
            st.dataframe(cfg['df_err'], use_container_width=True)

        st.subheader("Final Parameters Export")
        rows = []
        for model in models:
            res = fitter.results[model]
            rows.append({
                'model':     model,
                'B':         res['B'],
                'C_scaled':  res['C'],
                'A':         res['A'],
                'C_original': float(fitter.unscale_y(res['C'])),
                'sigma_mc':  float(fitter.y_range * res.get('sigma_mc', 0.0)),
                'sigma_lower': res.get('sigma_C_lower_unscaled', 0.0),
                'sigma_upper': res.get('sigma_C_upper_unscaled', 0.0),
                'ssr':        res['ssr'],
                'n_iter':     len(fitter.iteration_history[model]['params']),
            })
        df_export = pd.DataFrame(rows)
        st.dataframe(df_export, use_container_width=True)

        # CSV download
        csv_bytes = df_export.to_csv(index=False).encode()
        st.download_button(
            label="⬇ Download parameters as CSV",
            data=csv_bytes,
            file_name="varpro_results.csv",
            mime="text/csv",
        )


# ======================================================================
# Entry point
# ======================================================================
if __name__ == "__main__":
    main()