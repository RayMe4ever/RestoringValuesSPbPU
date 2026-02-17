import os
import pandas as pd
import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import requests

# ----------------------
#  Настройки
# ----------------------
BUSINESS_HTTP_BASE = os.getenv("BUSINESS_HTTP_BASE", "http://127.0.0.1:8000")

INSTALLATIONS = {
    "Установка 1": (8092, 8093),
    "Установка 2": (8094, 8095),
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

RECIEVER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Reciever")
BUSINESS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Business")

# кеш по mtime, чтобы перечитывать только когда файл реально обновился
_cache = {
    "raw_long": (None, None),
    "test_long": (None, None),
    "out_long": (None, None),
    "out": (None, None),
    "inp": (None, None),
    "metrics": (None, None),
}


def _read_csv_cached(key: str, path: str):
    if not os.path.exists(path):
        _cache[key] = (None, None)
        return None

    mtime = os.path.getmtime(path)
    last_mtime, last_df = _cache.get(key, (None, None))
    if last_mtime == mtime and last_df is not None:
        return last_df

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    _cache[key] = (mtime, df)
    return df


def get_feature_options(raw_port: int):
    path = os.path.join(RECIEVER_DIR, f"data_port_{raw_port}_long.csv")
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path, nrows=0)
        cols = [c for c in df.columns if c != "DateTime"]
        return [{"label": c, "value": c} for c in cols]
    except Exception:
        return []


# ----------------------
#  Layout
# ----------------------
app.layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H2("Панель управления", style={"marginBottom": "1rem", "color": "white"}),

            dbc.Label("Выберите установку", style={"color": "white"}),
            dcc.Dropdown(
                id="dropdown-installation",
                options=[{"label": k, "value": k} for k in INSTALLATIONS.keys()],
                value="Установка 1",
                clearable=False,
                style={"backgroundColor": "white", "color": "black"},
                className="mb-4"
            ),

            dbc.Label("Интервал интерполяции (мс)", style={"color": "white"}),
            dbc.Input(
                id="input-interval",
                type="number",
                min=100,
                step=100,
                value=5000,
                style={"backgroundColor": "#1A2138", "color": "white", "borderColor": "#444"}
            ),
            dbc.Button("Применить интервал", id="btn-apply-interval", color="primary", className="mt-2"),
            html.Div(id="apply-interval-msg", style={"marginTop": "0.5rem", "color": "#FFD700"}),

            html.Hr(style={"borderColor": "#444"}),

            dbc.Label("Выберите признак", style={"color": "white"}),
            dcc.Dropdown(
                id="dropdown-feature",
                options=[],
                value=None,
                clearable=False,
                style={"backgroundColor": "white", "color": "black"},
                className="mb-4"
            ),

            dbc.Label("Диапазон дат (необязательно)", style={"color": "white"}),
            dcc.DatePickerRange(
                id="date-picker",
                start_date=None,
                end_date=None,
                display_format="YYYY-MM-DD",
                style={"backgroundColor": "#1A2138", "color": "white"},
                className="mb-4"
            ),

            html.Div(id="status-message", style={"marginTop": "1rem", "color": "#FFD700", "minHeight": "1.5rem"}),

        ], width=3, style={"padding": "1rem", "backgroundColor": "#1A2138", "height": "100vh"}),

        dbc.Col([
            html.H4("Сырые данные", style={"color": "white"}),
            dcc.Graph(id="line-chart-raw", style={"height": "25vh"}),

            html.Hr(style={"borderColor": "#444"}),

            html.H4("Данные без пропусков", style={"color": "white", "marginTop": "1rem"}),
            dcc.Graph(id="line-chart-test", style={"height": "25vh"}),

            html.Hr(style={"borderColor": "#444", "marginTop": "1rem"}),

            html.H4("Заполнение пропусков", style={"color": "white", "marginTop": "1rem"}),
            dcc.Graph(id="line-out-long", style={"height": "25vh"}),

            html.Hr(style={"borderColor": "#444", "marginTop": "1rem"}),

            html.H4("Метрика работы модели", style={"color": "white", "marginTop": "1rem"}),
            html.Div(id="metrics-info", style={"color": "white", "marginBottom": "1rem"}),

            html.H4("Информация о записях", style={"color": "white", "marginTop": "1rem"}),
            html.Div(id="data-info", style={"color": "white", "marginBottom": "1rem"}),

            html.H4("Таблица обработки батча", style={"color": "white", "marginTop": "1rem"}),
            dash_table.DataTable(
                id="out-table",
                columns=[
                    {"name": "DateTime", "id": "DateTime"},
                    {"name": "Input", "id": "input"},
                    {"name": "Value", "id": "value"},
                ],
                data=[],
                page_size=10,
                style_header={"backgroundColor": "#1A2138", "color": "white"},
                style_cell={"backgroundColor": "#202946", "color": "white", "textAlign": "left"},
                style_table={"overflowX": "auto"},
            ),

            html.Hr(style={"borderColor": "#444", "marginTop": "1rem"}),

            html.H4("Таблица metrics (история метрик)", style={"color": "white", "marginTop": "1rem"}),
            dash_table.DataTable(
                id="metrics-file-table",
                columns=[],
                data=[],
                page_size=10,
                style_header={"backgroundColor": "#1A2138", "color": "white"},
                style_cell={"backgroundColor": "#202946", "color": "white", "textAlign": "left"},
                style_table={"overflowX": "auto"},
            ),

            dcc.Interval(id="interval-update", interval=2000, n_intervals=0),
        ], width=9, style={"padding": "1rem"}),
    ])
])


# ----------------------
#  Callback: опции признаков
# ----------------------
@app.callback(
    Output("dropdown-feature", "options"),
    Output("dropdown-feature", "value"),
    Input("dropdown-installation", "value"),
    Input("interval-update", "n_intervals"),
    State("dropdown-feature", "value"),
)
def update_feature_options(inst, _n, current_feature):
    raw_port, _ = INSTALLATIONS[inst]
    opts = get_feature_options(raw_port)
    values = [o["value"] for o in opts]
    if current_feature and current_feature in values:
        return opts, current_feature
    return opts, (values[0] if values else None)


# ----------------------
#  Callback: применить интервал (SYNC, без asyncio)
# ----------------------
@app.callback(
    Output("apply-interval-msg", "children"),
    Input("btn-apply-interval", "n_clicks"),
    State("input-interval", "value"),
)
def apply_interval_all_ports(n_clicks, new_interval):
    if not n_clicks:
        return ""
    try:
        r = requests.post(f"{BUSINESS_HTTP_BASE}/set_interval", json={"period_ms": int(new_interval)}, timeout=3)
        if r.ok:
            return f"Применено: интервал={new_interval} мс"
        return f"Ошибка: {r.status_code} {r.text}"
    except Exception as e:
        return f"Ошибка: {e}"


def _make_fig(x, y, title, color):
    return {
        "data": [{
            "x": x,
            "y": y,
            "type": "line",
            "name": title,
            "line": {"color": color},
        }],
        "layout": {
            "title": {"text": title, "font": {"color": "white"}},
            "paper_bgcolor": "#1A2138",
            "plot_bgcolor": "#202946",
            "font": {"color": "white"},
            "xaxis": {"color": "white", "gridcolor": "#444"},
            "yaxis": {"color": "white", "gridcolor": "#444"},
            "margin": {"l": 50, "r": 20, "t": 40, "b": 30},
        },
    }


# ----------------------
#  Callback: визуализация
# ----------------------
@app.callback(
    Output("line-chart-raw", "figure"),
    Output("line-chart-test", "figure"),
    Output("line-out-long", "figure"),
    Output("metrics-info", "children"),
    Output("data-info", "children"),
    Output("status-message", "children"),
    Output("out-table", "data"),
    Output("metrics-file-table", "columns"),
    Output("metrics-file-table", "data"),
    Input("interval-update", "n_intervals"),
    State("dropdown-installation", "value"),
    State("dropdown-feature", "value"),
    State("date-picker", "start_date"),
    State("date-picker", "end_date"),
)
def update_visualization(_n, inst, feature, start_date, end_date):
    raw_port, test_port = INSTALLATIONS[inst]

    raw_long_path = os.path.join(RECIEVER_DIR, f"data_port_{raw_port}_long.csv")
    test_long_path = os.path.join(RECIEVER_DIR, f"data_port_{test_port}_long.csv")

    inp_path = os.path.join(RECIEVER_DIR, f"data_port_{raw_port}.csv")
    out_path = os.path.join(BUSINESS_DIR, f"data_out_{raw_port}.csv")
    out_long_path = os.path.join(BUSINESS_DIR, f"data_out_{raw_port}_long.csv")

    metrics_path = os.path.join(BUSINESS_DIR, f"data_metrics_{test_port}.csv")

    df_raw_long = _read_csv_cached("raw_long", raw_long_path)
    if df_raw_long is None or df_raw_long.empty:
        return {}, {}, {}, "", "", f"Нет данных: {raw_long_path}", [], [], []

    if "DateTime" not in df_raw_long.columns:
        return {}, {}, {}, "", "", "Ошибка: нет DateTime в raw_long", [], [], []

    dff_raw = df_raw_long.copy()
    if start_date:
        dff_raw = dff_raw[dff_raw["DateTime"] >= start_date]
    if end_date:
        dff_raw = dff_raw[dff_raw["DateTime"] <= end_date]

    if not feature or feature not in dff_raw.columns or dff_raw.empty:
        return {}, {}, {}, "", "", "Нет данных для выбранного признака/диапазона", [], [], []

    fig_raw = _make_fig(dff_raw["DateTime"], dff_raw[feature], f"{inst} – сырые '{feature}'", "#FFD700")

    # test (без пропусков)
    fig_test = {
        "data": [],
        "layout": {"title": {"text": "Нет данных без пропусков", "font": {"color": "white"}},
                   "paper_bgcolor": "#1A2138", "plot_bgcolor": "#202946", "font": {"color": "white"}}
    }
    df_test_long = _read_csv_cached("test_long", test_long_path)
    if df_test_long is not None and not df_test_long.empty and "DateTime" in df_test_long.columns:
        dff_test = df_test_long.copy()
        if start_date:
            dff_test = dff_test[dff_test["DateTime"] >= start_date]
        if end_date:
            dff_test = dff_test[dff_test["DateTime"] <= end_date]
        if feature in dff_test.columns and not dff_test.empty:
            fig_test = _make_fig(dff_test["DateTime"], dff_test[feature], f"{inst} – без пропусков '{feature}'", "#1E90FF")

    # out_long (заполненные)
    fig_out_long = {
        "data": [],
        "layout": {"title": {"text": "Нет заполненных данных", "font": {"color": "white"}},
                   "paper_bgcolor": "#1A2138", "plot_bgcolor": "#202946", "font": {"color": "white"}}
    }
    df_out_long = _read_csv_cached("out_long", out_long_path)
    data_info = ""
    if df_out_long is not None and not df_out_long.empty and "DateTime" in df_out_long.columns:
        dff_out_long = df_out_long.copy()
        if start_date:
            dff_out_long = dff_out_long[dff_out_long["DateTime"] >= start_date]
        if end_date:
            dff_out_long = dff_out_long[dff_out_long["DateTime"] <= end_date]
        if feature in dff_out_long.columns and not dff_out_long.empty:
            fig_out_long = _make_fig(dff_out_long["DateTime"], dff_out_long[feature], f"{inst} – заполненные '{feature}'", "#FF901E")
            data_info = (
                f"Количество записей: {len(dff_out_long)}. "
                f"Первая дата: {dff_out_long['DateTime'].min()}. "
                f"Последняя дата: {dff_out_long['DateTime'].max()}."
            )

    # таблица input/value: merge по DateTime
    out_table_data = []
    df_out = _read_csv_cached("out", out_path)
    df_inp = _read_csv_cached("inp", inp_path)
    if df_out is not None and df_inp is not None and "DateTime" in df_out.columns and "DateTime" in df_inp.columns:
        if feature in df_out.columns and feature in df_inp.columns:
            m = df_inp[["DateTime", feature]].rename(columns={feature: "input"}).merge(
                df_out[["DateTime", feature]].rename(columns={feature: "value"}),
                on="DateTime",
                how="inner",
            )
            if start_date:
                m = m[m["DateTime"] >= start_date]
            if end_date:
                m = m[m["DateTime"] <= end_date]
            out_table_data = m.tail(10).to_dict("records")

    # metrics
    metrics_info = ""
    metrics_cols = []
    metrics_data = []
    df_metrics = _read_csv_cached("metrics", metrics_path)
    if df_metrics is not None and not df_metrics.empty:
        metrics_cols = [{"name": c, "id": c} for c in df_metrics.columns]
        metrics_data = df_metrics.to_dict("records")
        last = df_metrics.iloc[-1]
        metrics_info = ", ".join([f"{c} = {last[c]}" for c in df_metrics.columns])

    return (
        fig_raw,
        fig_test,
        fig_out_long,
        metrics_info,
        data_info,
        "",
        out_table_data,
        metrics_cols,
        metrics_data,
    )


if __name__ == "__main__":
    print("Запуск Dash-GUI (test)")
    app.run(debug=True, host="0.0.0.0", port=8050)