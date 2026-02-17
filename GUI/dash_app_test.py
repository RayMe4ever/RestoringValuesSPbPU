import os
import requests
import pandas as pd
from typing import Optional

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import aiohttp
import asyncio

# ----------------------
#  Константы и настройки
# ----------------------

BUSINESS_HTTP_BASE = "http://127.0.0.1:8000"

# «Концептуальные» установки с портами (raw, filled/test)
INSTALLATIONS = {
    "Установка 1": (8092, 8093),
    "Установка 2": (8094, 8095),
}

# ----------------------
#  Инициализация Dash
# ----------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

RECIEVER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Reciever")
BUSINESS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Business")

# ----------------------
#  Безопасное чтение CSV
# ----------------------

def safe_read_csv(path: str, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Читает CSV максимально устойчиво к гонкам записи.

    Возвращает None, если файл отсутствует/пустой/в процессе записи и не парсится.
    """
    try:
        if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
            return None
        if nrows is None:
            return pd.read_csv(path)
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        return None


# ----------------------
#  Вспомогательная функция: список признаков из «длинного» CSV
# ----------------------

def get_feature_options(raw_port: int):
    """
    Читает только шапку из Reciever/data_port_<raw_port>_long.csv (nrows=0),
    исключает 'DateTime' и возвращает [{"label":col,"value":col}, ...].
    Если файла нет или не удалось — [].
    """
    path = os.path.join(RECIEVER_DIR, f"data_port_{raw_port}_long.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, nrows=0)
            cols = [c for c in df.columns if c != "DateTime"]
            return [{"label": c, "value": c} for c in cols]
        except:
            return []
    return []

# ----------------------
#  Layout
# ----------------------

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
                dbc.Button(
                    "Применить интервал",
                    id="btn-apply-interval",
                    color="primary",
                    className="mt-2"
                ),
                html.Div(
                    id="apply-interval-msg",
                    style={"marginTop": "0.5rem", "color": "#FFD700"}
                ),
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

            html.Div(
                id="status-message",
                style={"marginTop": "1rem", "color": "#FFD700", "minHeight": "1.5rem"}
            ),

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


async def interval_send(new_interval):
    async with aiohttp.ClientSession() as session:
        async with session.post(
                f"{BUSINESS_HTTP_BASE}/set_interval",
                json={"period_ms": new_interval}
        ) as response:
            return response


@app.callback(
    Output("apply-interval-msg", "children"),
    Input("btn-apply-interval", "n_clicks"),
    State("input-interval", "value"),
)
def apply_interval_all_ports(n_clicks, new_interval):
    if not n_clicks:
        return ""
    try:
        r = asyncio.run(interval_send(new_interval))
        if r.ok:
            return f"Применено: интервал={new_interval} мс для всех портов"
        else:
            return "Ошибка при применении интервала"
    except Exception as e:
        return f"Ошибка: {e}"


# ----------------------
#  Стратегия чтения CSV для out-table
# ----------------------
#
# Вариант по умолчанию (самый надёжный для стриминга): *читаем CSV на каждом тике* dcc.Interval.
# Это убирает баг «залипания» данных из-за некорректного сравнения mtime(out) vs mtime(input).
#
# Альтернатива (если CSV очень большие): mtime-кэш. Для включения — раскомментируйте блок в секции #7.
#
df_out = None
df_input = None
# last_mtime_out = None
# last_mtime_in = None


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
    raw_port, filled_port = INSTALLATIONS[inst]

    raw_path = os.path.join(RECIEVER_DIR, f"data_port_{raw_port}_long.csv")
    input_path = os.path.join(RECIEVER_DIR, f"data_port_{raw_port}.csv")
    filled_path = os.path.join(RECIEVER_DIR, f"data_port_{filled_port}_long.csv")
    out_path_long = os.path.join(BUSINESS_DIR, f"data_out_{raw_port}_long.csv")

    # Для таблицы обработки батча предпочтительнее брать output по raw_port (результат заполнения пропусков).
    # Если такого snapshot-файла нет, падаем обратно на test/output по filled_port.
    out_path = os.path.join(BUSINESS_DIR, f"data_out_{raw_port}.csv")
    if not os.path.exists(out_path):
        out_path = os.path.join(BUSINESS_DIR, f"data_out_{filled_port}.csv")

    metrics_path = os.path.join(BUSINESS_DIR, f"data_metrics_{filled_port}.csv")

    if not os.path.exists(raw_path):
        return {}, {}, {}, "", "", "", [], [], []

    try:
        df_long = pd.read_csv(raw_path)
    except:
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

    fig_filled = {
        "data": [],
        "layout": {
            "title": {"text": "Нет заполненных данных", "font": {"color": "white"}},
            "paper_bgcolor": "#1A2138",
            "plot_bgcolor": "#202946",
            "font": {"color": "white"},
            "xaxis": {"color": "white", "gridcolor": "#444"},
            "yaxis": {"color": "white", "gridcolor": "#444"},
            "margin": {"l": 50, "r": 20, "t": 40, "b": 30}
        }
    }

    fig_out_long = {
        "data": [],
        "layout": {
            "title": {"text": "Нет заполненных данных", "font": {"color": "white"}},
            "paper_bgcolor": "#1A2138",
            "plot_bgcolor": "#202946",
            "font": {"color": "white"},
            "xaxis": {"color": "white", "gridcolor": "#444"},
            "yaxis": {"color": "white", "gridcolor": "#444"},
            "margin": {"l": 50, "r": 20, "t": 40, "b": 30}
        }
    }

    data_info = ""

    if os.path.exists(filled_path):
        try:
            df_filled = pd.read_csv(filled_path)
            dff_filled = df_filled.copy()
            if start_date:
                dff_filled = dff_filled[dff_filled["DateTime"] >= start_date]
            if end_date:
                dff_filled = dff_filled[dff_filled["DateTime"] <= end_date]

            if feature in dff_filled.columns:
                used_col = feature
                y_filled = dff_filled[feature]
            else:
                candidates = [c for c in dff_filled.columns if c != "DateTime"]
                used_col = candidates[0] if candidates else None
                y_filled = dff_filled[used_col] if used_col else []

            if not dff_filled.empty and used_col:
                fig_filled = {
                    "data": [{
                        "x": dff_filled["DateTime"],
                        "y": y_filled,
                        "type": "line",
                        "name": f"filled: {used_col}",
                        "line": {"color": "#1E90FF"}
                    }],
                    "layout": {
                        "title": {"text": f"{inst} – без пропусков '{used_col}'", "font": {"color": "white"}},
                        "paper_bgcolor": "#1A2138",
                        "plot_bgcolor": "#202946",
                        "font": {"color": "white"},
                        "xaxis": {"color": "white", "gridcolor": "#444"},
                        "yaxis": {"color": "white", "gridcolor": "#444"},
                        "margin": {"l": 50, "r": 20, "t": 40, "b": 30}
                    }
                }
                count = len(dff_filled)
                min_date = dff_filled["DateTime"].min()
                max_date = dff_filled["DateTime"].max()
                data_info = f"Количество записей: {count}. Первая дата: {min_date}. Последняя дата: {max_date}."
            else:
                data_info = "Нет обработанных данных"
        except:
            data_info = ""
    else:
        data_info = ""

    if os.path.exists(out_path_long):
        try:
            df_out_long = pd.read_csv(out_path_long)
            dff_out_long = df_out_long.copy()
            if start_date:
                dff_out_long = dff_out_long[dff_out_long["DateTime"] >= start_date]
            if end_date:
                dff_out_long = dff_out_long[dff_out_long["DateTime"] <= end_date]

            if feature in dff_out_long.columns:
                used_col = feature
                y_out = dff_out_long[feature]
            else:
                candidates = [c for c in dff_out_long.columns if c != "DateTime"]
                used_col = candidates[0] if candidates else None
                y_out = dff_out_long[used_col] if used_col else []

            if not dff_out_long.empty and used_col:
                fig_out_long = {
                    "data": [{
                        "x": dff_out_long["DateTime"],
                        "y": y_out,
                        "type": "line",
                        "name": f"filled: {used_col}",
                        "line": {"color": "#FF901E"}
                    }],
                    "layout": {
                        "title": {"text": f"{inst} – заполненные '{used_col}'", "font": {"color": "white"}},
                        "paper_bgcolor": "#1A2138",
                        "plot_bgcolor": "#202946",
                        "font": {"color": "white"},
                        "xaxis": {"color": "white", "gridcolor": "#444"},
                        "yaxis": {"color": "white", "gridcolor": "#444"},
                        "margin": {"l": 50, "r": 20, "t": 40, "b": 30}
                    }
                }
        except:
            pass

    # 7) Таблица out
    out_table_data = []
    if os.path.exists(out_path) and os.path.exists(input_path):
        try:
            df_out_cur = safe_read_csv(out_path)
            df_in_cur = safe_read_csv(input_path)

            # ===== Альтернатива: mtime-кэш (раскомментируйте при необходимости) =====
            # global df_out, df_input, last_mtime_out, last_mtime_in
            # m_out = os.path.getmtime(out_path)
            # m_in = os.path.getmtime(input_path)
            # if (df_out is None) or (df_input is None) or (last_mtime_out != m_out) or (last_mtime_in != m_in):
            #     df_out = pd.read_csv(out_path)
            #     df_input = pd.read_csv(input_path)
            #     last_mtime_out = m_out
            #     last_mtime_in = m_in
            # df_out_cur = df_out
            # df_in_cur = df_input

            if df_out_cur is None or df_in_cur is None:
                out_table_data = []
            else:
                dff_out = df_out_cur.copy()
                dff_in = df_in_cur.copy()

                if start_date and "DateTime" in dff_out.columns:
                    dff_out = dff_out[dff_out["DateTime"] >= start_date]
                if end_date and "DateTime" in dff_out.columns:
                    dff_out = dff_out[dff_out["DateTime"] <= end_date]

                if start_date and "DateTime" in dff_in.columns:
                    dff_in = dff_in[dff_in["DateTime"] >= start_date]
                if end_date and "DateTime" in dff_in.columns:
                    dff_in = dff_in[dff_in["DateTime"] <= end_date]

                if feature and feature in dff_out.columns:
                    out_col = feature
                else:
                    cols_out = [c for c in dff_out.columns if c != "DateTime"]
                    out_col = cols_out[0] if cols_out else None

                if out_col is None or "DateTime" not in dff_out.columns:
                    out_table_data = []
                else:
                    if feature and feature in dff_in.columns:
                        in_col = feature
                    elif out_col in dff_in.columns:
                        in_col = out_col
                    else:
                        cols_in = [c for c in dff_in.columns if c != "DateTime"]
                        in_col = cols_in[0] if cols_in else None

                    if ("DateTime" in dff_in.columns) and (in_col is not None):
                        merged = pd.merge(
                            dff_out[["DateTime", out_col]],
                            dff_in[["DateTime", in_col]],
                            on="DateTime",
                            how="left",
                        )
                        out_table_data = [
                            {"DateTime": r["DateTime"], "input": r.get(in_col), "value": r.get(out_col)}
                            for _, r in merged.iterrows()
                        ]
                    else:
                        n = min(len(dff_out), len(dff_in))
                        for i in range(n):
                            out_table_data.append(
                                {
                                    "DateTime": dff_out.iloc[i].get("DateTime", ""),
                                    "input": dff_in.iloc[i].get(in_col, None) if in_col else None,
                                    "value": dff_out.iloc[i].get(out_col, None),
                                }
                            )
        except Exception:
            out_table_data = []
    else:
        out_table_data = []

    metrics_file_columns = []
    metrics_file_data = []
    if os.path.exists(metrics_path):
        try:
            df_metrics = pd.read_csv(metrics_path)
            metrics_file_columns = [{"name": col, "id": col} for col in df_metrics.columns]
            metrics_file_data = df_metrics.to_dict("records")
        except:
            metrics_file_columns = []
            metrics_file_data = []

    metrics_info = ""
    if os.path.exists(metrics_path):
        try:
            dfm_full = pd.read_csv(metrics_path)
            if not dfm_full.empty:
                last = dfm_full.iloc[-1]
                parts = [f"{col} = {last[col]}" for col in dfm_full.columns]
                metrics_info = ", ".join(parts)
        except:
            metrics_info = ""

    status = ""

    return (
        fig_raw,
        fig_filled,
        fig_out_long,
        metrics_info,
        data_info,
        status,
        out_table_data,
        metrics_file_columns,
        metrics_file_data
    )


if __name__ == "__main__":
    print("Запуск Dash-GUI (Polling-CSV)")
    app.run(debug=False, host="0.0.0.0", port=8050)
