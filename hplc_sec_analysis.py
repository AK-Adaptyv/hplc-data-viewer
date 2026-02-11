"""
HPLC-SEC Chromatogram Analysis — Interactive Dash App

Usage:
    pip install dash plotly pandas numpy scipy
    python hplc_sec_analysis.py
    Open http://127.0.0.1:8050

Set DATA_ROOT env var to override the default chromatogram directory.
Default on Windows:  C:/CDSProjects/hplc_test_ak/csv_chromatograms
Default elsewhere:   scans working directory for .rslt folders
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import plotly.graph_objects as go

# ── Color configuration ────────────────────────────────────────────────────

WAVELENGTH_COLORS = {
    "220 nm": {"base": "#8424F2", "peak": "#DE63FF"},
    "280 nm": {"base": "#2724F2", "peak": "#3633FF"},
    "395 nm": {"base": "#42AEFC", "peak": "#76E6F5"},
}

ALL_WAVELENGTHS = ["220 nm", "280 nm", "395 nm"]


def hex_to_rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def rgb_lerp(c1, c2, t):
    """Linearly interpolate two hex colors, return 'rgb(r,g,b)' string."""
    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"rgb({r},{g},{b})"


def compute_glow(time, peaks, sigma=0.08):
    """Return array 0-1 indicating proximity to any peak maximum (Gaussian)."""
    glow = np.zeros(len(time))
    for p in peaks:
        glow = np.maximum(glow, np.exp(-0.5 * ((time - p["rt"]) / sigma) ** 2))
    return glow


# ── Data root configuration ────────────────────────────────────────────────

WINDOWS_DEFAULT = r"C:/CDSProjects/hplc_test_ak/csv_chromatograms"


def resolve_data_root():
    """Determine the root directory containing sequence folders."""
    # 1. Explicit env var
    env = os.environ.get("DATA_ROOT")
    if env and os.path.isdir(env):
        return env
    # 2. Windows default
    if sys.platform == "win32" and os.path.isdir(WINDOWS_DEFAULT):
        return WINDOWS_DEFAULT
    # 3. Fallback: current working directory
    return os.getcwd()


DATA_ROOT = resolve_data_root()

# ── Data loading ────────────────────────────────────────────────────────────

CHANNEL_MAP = {
    "DAD1A": "280 nm",
    "DAD1B": "220 nm",
    "DAD1C": "395 nm",
}


def _has_csv_samples(folder):
    """Check if a folder contains at least one subdirectory with .CSV files."""
    for sub in os.listdir(folder):
        sub_path = os.path.join(folder, sub)
        if os.path.isdir(sub_path) and glob.glob(os.path.join(sub_path, "*.CSV")):
            return True
    return False


def list_sequences(data_root):
    """Return sorted list of sequence folder names inside data_root."""
    seqs = []
    for entry in sorted(os.listdir(data_root)):
        if entry.startswith("."):
            continue
        full = os.path.join(data_root, entry)
        if os.path.isdir(full) and _has_csv_samples(full):
            seqs.append(entry)
    return seqs


def list_samples(data_root, sequence):
    """Return sorted list of sample folder names inside a sequence."""
    seq_path = os.path.join(data_root, sequence)
    if not os.path.isdir(seq_path):
        return []
    return sorted(
        name for name in os.listdir(seq_path)
        if os.path.isdir(os.path.join(seq_path, name))
    )


def load_sample(data_root, sequence, sample):
    """Load a single sample → dict[wavelength_label] = DataFrame(time, response)."""
    sample_path = os.path.join(data_root, sequence, sample)
    if not os.path.isdir(sample_path):
        return {}
    result = {}
    for csv_path in glob.glob(os.path.join(sample_path, "*.CSV")):
        basename = os.path.basename(csv_path)
        for channel_key, wl_label in CHANNEL_MAP.items():
            if channel_key in basename:
                df = pd.read_csv(csv_path, header=None, names=["time", "response"])
                result[wl_label] = df
                break
    return result


# ── Peak detection & integration ────────────────────────────────────────────

def detect_peaks(df, prominence=0.5, min_width=5, min_height=0.0):
    """Detect peaks and compute area, %area, retention time, and estimated MW."""
    response = df["response"].values
    time = df["time"].values

    peaks, _ = find_peaks(
        response, prominence=prominence, width=min_width, height=min_height,
    )
    if len(peaks) == 0:
        return []

    widths_result = peak_widths(response, peaks, rel_height=1.0)
    left_ips = widths_result[2]
    right_ips = widths_result[3]
    _trapz = getattr(np, "trapezoid", np.trapz)

    peak_list = []
    for i, peak_idx in enumerate(peaks):
        li = max(int(np.floor(left_ips[i])), 0)
        ri = min(int(np.ceil(right_ips[i])) + 1, len(time))

        area = float(_trapz(np.maximum(response[li:ri], 0), time[li:ri]))
        rt = float(time[peak_idx])
        mw = 10 ** (-0.8316 * rt + 8.517)

        peak_list.append({
            "peak_idx": int(peak_idx),
            "left_idx": li,
            "right_idx": ri - 1,
            "rt": round(rt, 3),
            "height": round(float(response[peak_idx]), 3),
            "area": round(area, 4),
            "mw_da": mw,
            "mw_kda": round(mw / 1000, 2),
        })

    total_area = sum(p["area"] for p in peak_list)
    for p in peak_list:
        p["area_pct"] = round(100 * p["area"] / total_area, 2) if total_area > 0 else 0.0

    return peak_list


# ── Dash app ────────────────────────────────────────────────────────────────

app = Dash(__name__, suppress_callback_exceptions=True)

app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body, #react-entry-point, #_dash-app-content {
                background-color: #000000 !important;
                margin: 0;
                padding: 0;
                font-size: 20px;
            }

            /* ── Dark dropdowns (React-Select) ── */
            .Select-control,
            .Select-menu-outer,
            .Select-menu,
            .Select-option,
            .Select-value,
            .Select-placeholder,
            .Select-input,
            .Select-input > input,
            .Select--single > .Select-control .Select-value,
            .VirtualizedSelectOption,
            .VirtualizedSelectFocusedOption,
            .dash-dropdown .Select-control,
            .dash-dropdown .Select-menu-outer {
                background-color: #111 !important;
                color: #E0E0E0 !important;
                border-color: #333 !important;
                font-size: 18px !important;
            }
            .Select-control { min-height: 44px !important; }
            .Select-arrow-zone .Select-arrow {
                border-top-color: #888 !important;
            }
            .Select-option.is-focused,
            .VirtualizedSelectFocusedOption {
                background-color: #222 !important;
            }
            .Select-option.is-selected {
                background-color: #333 !important;
            }
            .Select-value-label {
                color: #E0E0E0 !important;
                font-size: 18px !important;
            }
            .Select-noresults {
                background-color: #111 !important;
                color: #888 !important;
            }

            /* ── Slider track/handle dark ── */
            .rc-slider-track { background-color: #555 !important; }
            .rc-slider-rail  { background-color: #222 !important; }
            .rc-slider-handle {
                border-color: #888 !important;
                background-color: #333 !important;
                width: 18px !important;
                height: 18px !important;
                margin-top: -7px !important;
            }
            .rc-slider-dot   { background-color: #333 !important; border-color: #555 !important; }
            .rc-slider-mark-text { color: #888 !important; font-size: 15px !important; }
            .rc-slider-tooltip-inner { font-size: 15px !important; }

            /* ── Scale up ── */
            body { font-size: 20px; }
            .dash-table-container { font-size: 18px !important; }

            /* ── Gradient title ── */
            .gradient-title {
                text-align: center;
                font-size: 2.8rem;
                font-weight: bold;
                background: linear-gradient(90deg, #27ADF5, #FAABFF);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 4px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

INITIAL_SEQUENCES = list_sequences(DATA_ROOT)

app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "maxWidth": "1400px",
        "margin": "0 auto",
        "padding": "30px 40px",
        "backgroundColor": "#000000",
        "color": "#E0E0E0",
        "minHeight": "100vh",
        "fontSize": "20px",
    },
    children=[
        html.H2("HPLC-SEC Chromatogram Analysis", className="gradient-title"),
        html.P(f"Data root: {DATA_ROOT}", style={"color": "#888", "fontSize": "1em", "textAlign": "center"}),

        # ── Controls row ───────────────────────────────────────────────
        html.Div(
            style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "alignItems": "flex-end"},
            children=[
                html.Div([
                    html.Label("Sequence", style={"color": "#CCC", "fontSize": "1.15em", "marginBottom": "6px"}),
                    dcc.Dropdown(
                        id="sequence-dropdown",
                        options=[{"label": s, "value": s} for s in INITIAL_SEQUENCES],
                        value=INITIAL_SEQUENCES[0] if INITIAL_SEQUENCES else None,
                        style={"width": "480px"},
                    ),
                ]),
                html.Div([
                    html.Label("Sample", style={"color": "#CCC", "fontSize": "1.15em", "marginBottom": "6px"}),
                    dcc.Dropdown(id="sample-dropdown", style={"width": "340px"}),
                ]),
                html.Div([
                    html.Label("Wavelength", style={"color": "#CCC", "fontSize": "1.15em", "marginBottom": "6px"}),
                    dcc.Dropdown(
                        id="wavelength-dropdown",
                        options=[{"label": w, "value": w} for w in ALL_WAVELENGTHS],
                        value="280 nm",
                        style={"width": "180px"},
                    ),
                ]),
                html.Div(style={"marginBottom": "4px"}, children=[
                    html.Button(
                        "Refresh sequences",
                        id="refresh-btn",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#222",
                            "color": "#CCC",
                            "border": "1px solid #444",
                            "padding": "10px 18px",
                            "cursor": "pointer",
                            "fontSize": "0.95em",
                            "borderRadius": "4px",
                        },
                    ),
                ]),
            ],
        ),

        # ── Sliders row ───────────────────────────────────────────────
        html.Div(
            style={"display": "flex", "gap": "40px", "flexWrap": "wrap", "marginTop": "16px"},
            children=[
                html.Div(style={"flex": "1", "minWidth": "240px"}, children=[
                    html.Label(id="prominence-label", children="Prominence: 0.5 mAu", style={"color": "#CCC", "fontSize": "1.1em"}),
                    dcc.Slider(
                        id="prominence-slider",
                        min=0.01, max=10, step=0.01, value=0.5,
                        marks={0.01: "0.01", 1: "1", 5: "5", 10: "10"},
                        tooltip={"placement": "bottom"},
                    ),
                ]),
                html.Div(style={"flex": "1", "minWidth": "240px"}, children=[
                    html.Label(id="width-label", children="Min width: 5 pts", style={"color": "#CCC", "fontSize": "1.1em"}),
                    dcc.Slider(
                        id="width-slider",
                        min=1, max=50, step=1, value=5,
                        marks={1: "1", 10: "10", 25: "25", 50: "50"},
                        tooltip={"placement": "bottom"},
                    ),
                ]),
                html.Div(style={"flex": "1", "minWidth": "240px"}, children=[
                    html.Label(id="height-label", children="Min height: 0 mAu", style={"color": "#CCC", "fontSize": "1.1em"}),
                    dcc.Slider(
                        id="height-slider",
                        min=0, max=50, step=0.1, value=0,
                        marks={0: "0", 10: "10", 25: "25", 50: "50"},
                        tooltip={"placement": "bottom"},
                    ),
                ]),
            ],
        ),

        # ── Chromatogram plot ──────────────────────────────────────────
        dcc.Graph(id="chromatogram", style={"marginTop": "20px"}, clear_on_unhover=True),

        # ── Peak table ─────────────────────────────────────────────────
        html.H4("Detected Peaks", style={"marginTop": "16px", "color": "#FFFFFF", "fontSize": "1.4em"}),
        dash_table.DataTable(
            id="peak-table",
            columns=[
                {"name": "#", "id": "num"},
                {"name": "RT (min)", "id": "rt"},
                {"name": "Height (mAu)", "id": "height"},
                {"name": "Area (mAu·min)", "id": "area"},
                {"name": "Area %", "id": "area_pct"},
                {"name": "Est. MW (kDa)", "id": "mw_kda"},
            ],
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "center",
                "padding": "12px 18px",
                "fontSize": "17px",
                "fontFamily": "Arial, sans-serif",
                "backgroundColor": "#111111",
                "color": "#E0E0E0",
                "border": "1px solid #333",
            },
            style_header={
                "fontWeight": "bold",
                "fontSize": "18px",
                "backgroundColor": "#1A1A1A",
                "color": "#FFFFFF",
                "border": "1px solid #333",
            },
        ),
    ],
)


# ── Callbacks ───────────────────────────────────────────────────────────────

@callback(
    Output("sequence-dropdown", "options"),
    Input("refresh-btn", "n_clicks"),
)
def refresh_sequences(_n):
    """Re-scan DATA_ROOT for sequence folders (also runs on initial load)."""
    seqs = list_sequences(DATA_ROOT)
    return [{"label": s, "value": s} for s in seqs]


@callback(
    Output("sample-dropdown", "options"),
    Output("sample-dropdown", "value"),
    Input("sequence-dropdown", "value"),
)
def update_samples(sequence):
    """Populate sample dropdown when a sequence is selected."""
    if not sequence:
        return [], None
    samples = list_samples(DATA_ROOT, sequence)
    options = [{"label": s, "value": s} for s in samples]
    default = samples[0] if samples else None
    return options, default


@callback(
    Output("prominence-label", "children"),
    Output("width-label", "children"),
    Output("height-label", "children"),
    Input("prominence-slider", "value"),
    Input("width-slider", "value"),
    Input("height-slider", "value"),
)
def update_slider_labels(prom, wid, hgt):
    return (
        f"Prominence: {prom} mAu",
        f"Min width: {wid} pts",
        f"Min height: {hgt} mAu",
    )


N_COLOR_LEVELS = 20


def _build_gradient_traces(time, response, glow, base_hex, peak_hex, wavelength):
    """Split the chromatogram into segments coloured by glow level."""
    levels = np.clip((glow * N_COLOR_LEVELS).astype(int), 0, N_COLOR_LEVELS)
    traces = []
    i = 0
    first = True
    while i < len(time) - 1:
        lvl = levels[i]
        j = i + 1
        while j < len(time) and levels[j] == lvl:
            j += 1
        end = min(j + 1, len(time))
        color = rgb_lerp(base_hex, peak_hex, lvl / N_COLOR_LEVELS)
        traces.append(go.Scatter(
            x=time[i:end],
            y=response[i:end],
            mode="lines",
            line={"color": color, "width": 4},
            name=wavelength if first else None,
            showlegend=first,
            legendgroup="chromatogram",
            hovertemplate="Time: %{x:.3f} min<br>Response: %{y:.3f} mAu<extra></extra>",
        ))
        first = False
        i = j
    return traces


@callback(
    Output("chromatogram", "figure"),
    Output("peak-table", "data"),
    Input("sequence-dropdown", "value"),
    Input("sample-dropdown", "value"),
    Input("wavelength-dropdown", "value"),
    Input("prominence-slider", "value"),
    Input("width-slider", "value"),
    Input("height-slider", "value"),
)
def update_plot(sequence, sample, wavelength, prominence, min_width, min_height):
    fig = go.Figure()
    table_data = []

    # Apply dark layout even when no data
    dark_layout = dict(
        hovermode="closest",
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font={"family": "Arial, sans-serif", "color": "#CCCCCC", "size": 16},
        margin={"t": 60, "b": 60, "l": 70, "r": 30},
        height=650,
        hoverlabel={"font_size": 16},
        xaxis={"gridcolor": "#222", "zerolinecolor": "#333", "title_font_size": 18, "tickfont_size": 15},
        yaxis={"gridcolor": "#222", "zerolinecolor": "#333", "title_font_size": 18, "tickfont_size": 15},
        legend={"font_size": 16},
    )

    if not sequence or not sample or not wavelength:
        fig.update_layout(title="Select a sequence, sample, and wavelength", **dark_layout)
        return fig, table_data

    # Load data on demand
    sample_data = load_sample(DATA_ROOT, sequence, sample)
    if wavelength not in sample_data:
        fig.update_layout(title=f"No {wavelength} data for {sample}", **dark_layout)
        return fig, table_data

    df = sample_data[wavelength]
    time = df["time"].values
    response = df["response"].values

    colors = WAVELENGTH_COLORS.get(wavelength, {"base": "#FFFFFF", "peak": "#FFFFFF"})
    base_hex = colors["base"]
    peak_hex = colors["peak"]

    # Detect peaks
    peaks = detect_peaks(df, prominence=prominence, min_width=min_width, min_height=min_height)

    # Gradient chromatogram line
    glow = compute_glow(time, peaks) if peaks else np.zeros(len(time))
    for tr in _build_gradient_traces(time, response, glow, base_hex, peak_hex, wavelength):
        fig.add_trace(tr)

    # Shaded peak areas
    fill_r, fill_g, fill_b = hex_to_rgb(peak_hex)
    fill_dim = f"rgba({fill_r},{fill_g},{fill_b},0.12)"
    fill_bright = f"rgba({fill_r},{fill_g},{fill_b},0.38)"
    for p in peaks:
        li, ri = p["left_idx"], p["right_idx"] + 1
        seg_t = time[li:ri]
        seg_r = response[li:ri]
        hover_txt = (
            f"RT: {p['rt']} min<br>"
            f"Area: {p['area']} mAu·min<br>"
            f"Area%: {p['area_pct']}%<br>"
            f"MW: {p['mw_kda']} kDa"
        )
        fig.add_trace(go.Scatter(
            x=np.concatenate([seg_t, seg_t[::-1]]),
            y=np.concatenate([seg_r, np.zeros(len(seg_r))]),
            fill="toself",
            fillcolor=fill_dim,
            line={"width": 0},
            showlegend=False,
            hoverinfo="text",
            hovertext=hover_txt,
            meta={"peak_fill": True, "fill_dim": fill_dim, "fill_bright": fill_bright},
        ))

    # Peak markers
    if peaks:
        marker_x = [p["rt"] for p in peaks]
        marker_y = [p["height"] for p in peaks]
        hover_texts = [
            f"RT: {p['rt']} min<br>"
            f"Area: {p['area']} mAu·min<br>"
            f"Area%: {p['area_pct']}%<br>"
            f"MW: {p['mw_kda']} kDa"
            for p in peaks
        ]
        fig.add_trace(go.Scatter(
            x=marker_x, y=marker_y,
            mode="markers+text",
            marker={"size": 12, "color": peak_hex, "symbol": "diamond",
                     "line": {"width": 1.5, "color": "#FFFFFF"}},
            text=[f"{p['rt']}" for p in peaks],
            textposition="top center",
            textfont={"size": 14, "color": "#FFFFFF"},
            name="Peaks",
            hovertext=hover_texts,
            hoverinfo="text",
        ))

    fig.update_layout(
        title={"text": f"{sample} — {wavelength}", "font": {"color": "#FFFFFF", "size": 22}},
        xaxis_title="Retention Time (min)",
        yaxis_title="Response (mAu)",
        **dark_layout,
    )

    for i, p in enumerate(peaks):
        table_data.append({
            "num": i + 1,
            "rt": p["rt"],
            "height": p["height"],
            "area": p["area"],
            "area_pct": p["area_pct"],
            "mw_kda": p["mw_kda"],
        })

    return fig, table_data


# ── Clientside callback: brighten peak area on hover ────────────────────────

app.clientside_callback(
    """
    function(hoverData, figure) {
        if (!figure || !figure.data) return window.dash_clientside.no_update;
        // Don't interfere when hover clears (e.g. after figure update)
        if (!hoverData) return window.dash_clientside.no_update;
        var newFig = JSON.parse(JSON.stringify(figure));
        var hoveredTrace = null;
        if (hoverData.points && hoverData.points.length > 0) {
            hoveredTrace = hoverData.points[0].curveNumber;
        }
        for (var i = 0; i < newFig.data.length; i++) {
            var tr = newFig.data[i];
            if (tr.meta && tr.meta.peak_fill) {
                if (i === hoveredTrace) {
                    tr.fillcolor = tr.meta.fill_bright;
                } else {
                    tr.fillcolor = tr.meta.fill_dim;
                }
            }
        }
        return newFig;
    }
    """,
    Output("chromatogram", "figure", allow_duplicate=True),
    Input("chromatogram", "hoverData"),
    State("chromatogram", "figure"),
    prevent_initial_call=True,
)


# ── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    seqs = list_sequences(DATA_ROOT)
    print(f"Data root: {DATA_ROOT}")
    print(f"Found {len(seqs)} sequence(s): {', '.join(seqs) if seqs else '(none)'}")
    app.run(debug=False, host="0.0.0.0", port=8050)
