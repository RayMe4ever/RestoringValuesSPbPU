# GUI/dash_app_test.py
import os
import sys
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from logging_setup import setup_logging, new_op_id

import requests
import pandas as pd

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc

logger = setup_logging("dash_test")

BUSINESS_HTTP_BASE = os.getenv("BUSINESS_HTTP_BASE", "http://127.0.0.1:8000")

INSTALLATIONS = {
    "Установка 1": (8092, 8093),
    "Установка 2": (8094, 8095),
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

RECIEVER_DIR = os.path.join(PROJECT_ROOT, "Reciever")
BUSINESS_DIR = os.path.join(PROJECT_ROOT, "Business")


@server.route("/healthz")
def healthz():
    op_id = new_op_id("health")
    logger.info("healthz ok", extra={"op_id": op_id})
    return {"status": "ok", "service": "dash_test", "time_utc": datetime.now(timezone.utc).isoformat(), "op_id": op_id}


def get_feature_options(raw_port: int):
    path = os.path.join(RECIEVER_DIR, f"data_port_{raw_port}_long.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, nrows=0)
            cols = [c for c in df.columns if c != "DateTime"]
            return [{"label": c, "value": c} for c in cols]
        except Exception as e:
            logger.warning("failed read header path=%s err=%s", path, e, extra={"op_id": new_op_id("hdr")})
            return []
    return []


app.layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H2("Панель управления", style={"marginBottom": "1rem", "color": "white"}),

            html.Div([
                dbc.Label("Выберите установку", style={"color": "white"}),
                dcc.Dropdown(
                    id="dropdown-installation",
                    options=[{"label": k, "value": k} for k in INSTALLATIONS.keys()],
                    value="Установка 1",
                    clearable=False,
                    style={"backgroundColor": "white", "color": "black"},
                    className="mb-4"
                ),
            ]),

            html.Div([
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
            ], className="mb-4"),

            html.Div([
                dbc.Label("Выберите признак", style={"color": "white"}),
                dcc.Dropdown(
                    id="dropdown-feature",
                    options=[],
                    value=None,
                    clearable=False,
                    style={"backgroundColor": "white", "color": "black"},
                    className="mb-4"
                ),
            ]),

            html.Div([
                dbc.Label("Диапазон дат (необязательно)", style={"color": "white"}),
                dcc.DatePickerRange(
                    id="date-picker",
                    start_date=None,
                    end_date=None,
                    display_format="YYYY-MM-DD",
                    style={"backgroundColor": "#1A2138", "color": "white"},
                    className="mb-4"
                ),
            ]),

            html.Div(id="status-message", style={"marginTop": "1rem", "color": "#FFD700", "minHeight": "1.5rem"}),

        ], width=3, style={"padding": "1rem", "backgroundColor": "#1A2138", "height": "100vh"}),

        dbc.Col([
            html.H4("Сырые данные", style={"color": "white"}),
            dcc.Graph(id="line-chart-raw", style={"height": "25vh"}),

            html.Hr(style={"borderColor": "#444"}),

            html.H4("Данные без пропусков", style={"color": "white", "marginTop": "1rem"}),
            dcc.Graph(id="line-chart-filled", style={"height": "25vh"}),

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


@app.callback(
    Output("dropdown-feature", "options"),
    Output("dropdown-feature", "value"),
    Input("dropdown-installation", "value"),
    Input("interval-update", "n_intervals"),
    State("dropdown-feature", "value"),
)
def update_feature_options(inst, n_intervals, current_feature):
    raw_port, _ = INSTALLATIONS[inst]
    opts = get_feature_options(raw_port)
    values = [opt["value"] for opt in opts]

    if current_feature and current_feature in values:
        return opts, current_feature

    default = values[0] if values else None
    return opts, default


@app.callback(
    Output("apply-interval-msg", "children"),
    Input("btn-apply-interval", "n_clicks"),
    State("input-interval", "value"),
)
def apply_interval_all_ports(n_clicks, new_interval):
    if not n_clicks:
        return ""
    op_id = new_op_id("ui-interval")
    try:
        r = requests.post(
            f"{BUSINESS_HTTP_BASE}/set_interval",
            json={"period_ms": int(new_interval)},
            timeout=5,
        )
        logger.info(
            "apply interval period_ms=%s status_code=%s",
            new_interval,
            r.status_code,
            extra={"op_id": op_id},
        )
        if r.ok:
            return f"Применено: интервал={new_interval} мс для всех портов"
        return f"Ошибка при применении интервала: HTTP {r.status_code}"
    except Exception as e:
        logger.exception("apply interval failed err=%s", e, extra={"op_id": op_id})
        return f"Ошибка: {e}"


df_out = None
df_input = None


@app.callback(
    Output("line-chart-raw", "figure"),
    Output("line-chart-filled", "figure"),
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
def update_visualization(n_intervals, inst, feature, start_date, end_date):
    op_id = new_op_id(f"ui-{n_intervals}")
    raw_port, filled_port = INSTALLATIONS[inst]

    raw_path = os.path.join(RECIEVER_DIR, f"data_port_{raw_port}_long.csv")
    input_path = os.path.join(RECIEVER_DIR, f"data_port_{raw_port}.csv")
    filled_path = os.path.join(RECIEVER_DIR, f"data_port_{filled_port}_long.csv")
    out_path_long = os.path.join(BUSINESS_DIR, f"data_out_{raw_port}_long.csv")
    out_path = os.path.join(BUSINESS_DIR, f"data_out_{filled_port}.csv")
    metrics_path = os.path.join(BUSINESS_DIR, f"data_metrics_{filled_port}.csv")

    if not os.path.exists(raw_path):
        logger.debug("raw_long missing path=%s", raw_path, extra={"op_id": op_id})
        return {}, {}, {}, "", "", "", [], [], []

    try:
        df_long = pd.read_csv(raw_path)
    except Exception as e:
        logger.warning("csv read failed path=%s err=%s", raw_path, e, extra={"op_id": op_id})
        return {}, {}, {}, "", "", "Ошибка при чтении CSV", [], [], []

    if df_long["DateTime"].isnull().all():
        return {}, {}, {}, "", "", "Потеря соединения с установкой", [], [], []

    dff_raw = df_long.copy()
    if start_date:
        dff_raw = dff_raw[dff_raw["DateTime"] >= start_date]
    if end_date:
        dff_raw = dff_raw[dff_raw["DateTime"] <= end_date]

    if not feature or feature not in dff_raw.columns or dff_raw.empty:
        return {}, {}, {}, "", "", "Нет данных для выбранного признака/диапазона", [], [], []

    fig_raw = {
        "data": [{
            "x": dff_raw["DateTime"],
            "y": dff_raw[feature],
            "type": "line",
            "name": f"raw: {feature}",
            "line": {"color": "#FFD700"}
        }],
        "layout": {
            "title": {"text": f"{inst} – сырые '{feature}'", "font": {"color": "white"}},
            "paper_bgcolor": "#1A2138",
            "plot_bgcolor": "#202946",
            "font": {"color": "white"},
            "xaxis": {"color": "white", "gridcolor": "#444"},
            "yaxis": {"color": "white", "gridcolor": "#444"},
            "margin": {"l": 50, "r": 20, "t": 40, "b": 30}
        }
    }

    fig_filled = {"data": [], "layout": {"title": {"text": "Нет заполненных данных", "font": {"color": "white"}}}}
    fig_out_long = {"data": [], "layout": {"title": {"text": "Нет заполненных данных", "font": {"color": "white"}}}}

    data_info = ""
    out_table_data = []

    # Filled поток
    if os.path.exists(filled_path):
        try:
            df_filled = pd.read_csv(filled_path)
            dff_filled = df_filled.copy()
            if start_date:
                dff_filled = dff_filled[dff_filled["DateTime"] >= start_date]
            if end_date:
                dff_filled = dff_filled[dff_filled["DateTime"] <= end_date]

            used_col = feature if feature in dff_filled.columns else next((c for c in dff_filled.columns if c != "DateTime"), None)
            if used_col and not dff_filled.empty:
                fig_filled = {
                    "data": [{
                        "x": dff_filled["DateTime"],
                        "y": dff_filled[used_col],
                        "type": "line",
                        "name": f"filled: {used_col}",
                        "line": {"color": "#1E90FF"}
                    }],
                    "layout": {"title": {"text": f"{inst} – без пропусков '{used_col}'", "font": {"color": "white"}}}
                }
        except Exception as e:
            logger.warning("filled csv read failed err=%s", e, extra={"op_id": op_id})

    # Out_long поток
    if os.path.exists(out_path_long):
        try:
            df_out_long = pd.read_csv(out_path_long)
            dff_out_long = df_out_long.copy()
            if start_date:
                dff_out_long = dff_out_long[dff_out_long["DateTime"] >= start_date]
            if end_date:
                dff_out_long = dff_out_long[dff_out_long["DateTime"] <= end_date]

            used_col = feature if feature in dff_out_long.columns else next((c for c in dff_out_long.columns if c != "DateTime"), None)
            if used_col and not dff_out_long.empty:
                fig_out_long = {
                    "data": [{
                        "x": dff_out_long["DateTime"],
                        "y": dff_out_long[used_col],
                        "type": "line",
                        "name": f"filled: {used_col}",
                        "line": {"color": "#FF901E"}
                    }],
                    "layout": {"title": {"text": f"{inst} – заполненные '{used_col}'", "font": {"color": "white"}}}
                }

                count = len(dff_out_long)
                min_date = dff_out_long["DateTime"].min()
                max_date = dff_out_long["DateTime"].max()
                data_info = f"Количество записей: {count}. Первая дата: {min_date}. Последняя дата: {max_date}."
        except Exception as e:
            logger.warning("out_long csv read failed err=%s", e, extra={"op_id": op_id})

    # Таблица out
    if os.path.exists(out_path) and os.path.exists(input_path):
        try:
            global df_out, df_input
            if (df_out is None) or (df_input is None) or (os.path.getmtime(out_path) >= os.path.getmtime(input_path)):
                df_out = pd.read_csv(out_path)
                df_input = pd.read_csv(input_path)

            dff_out = df_out.copy()
            dff_input = df_input.copy()

            if start_date:
                dff_out = dff_out[dff_out["DateTime"] >= start_date]
                dff_input = dff_input[dff_input["DateTime"] >= start_date]
            if end_date:
                dff_out = dff_out[dff_out["DateTime"] <= end_date]
                dff_input = dff_input[dff_input["DateTime"] <= end_date]

            if feature in dff_out.columns and feature in dff_input.columns:
                for i in range(min(len(dff_out), len(dff_input))):
                    out_table_data.append({
                        "DateTime": dff_out.iloc[i]["DateTime"],
                        "input": dff_input.iloc[i][feature],
                        "value": dff_out.iloc[i][feature]
                    })
        except Exception as e:
            logger.warning("out table build failed err=%s", e, extra={"op_id": op_id})

    # История метрик
    metrics_file_columns = []
    metrics_file_data = []
    if os.path.exists(metrics_path):
        try:
            df_metrics = pd.read_csv(metrics_path)
            metrics_file_columns = [{"name": col, "id": col} for col in df_metrics.columns]
            metrics_file_data = df_metrics.to_dict("records")
        except Exception as e:
            logger.warning("metrics csv read failed err=%s", e, extra={"op_id": op_id})

    # Последняя метрика
    metrics_info = ""
    if os.path.exists(metrics_path):
        try:
            dfm_full = pd.read_csv(metrics_path)
            if not dfm_full.empty:
                last = dfm_full.iloc[-1]
                metrics_info = ", ".join([f"{col} = {last[col]}" for col in dfm_full.columns])
        except Exception:
            metrics_info = ""

    return (
        fig_raw,
        fig_filled,
        fig_out_long,
        metrics_info,
        data_info,
        "",
        out_table_data,
        metrics_file_columns,
        metrics_file_data,
    )


if __name__ == "__main__":
    dash_host = os.getenv("DASH_HOST") or os.getenv("HOST") or "0.0.0.0"
    dash_port = int(os.getenv("DASH_PORT") or os.getenv("PORT") or "8050")
    logger.info("starting dash host=%s port=%s business=%s", dash_host, dash_port, BUSINESS_HTTP_BASE, extra={"op_id": new_op_id("dash-start")})
    app.run(debug=True, host=dash_host, port=dash_port)
