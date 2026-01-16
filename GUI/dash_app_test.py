import os, socket
import pandas as pd
import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import aiohttp
import asyncio

# ----------------------
#  Константы и настройки
# ----------------------

BUSINESS_HTTP_BASE = "http://localhost:8000"

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

RECIEVER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Reciever", "received_data")
BUSINESS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Business")
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
        # ===========================================
        #  Левое меню: выбор установки, интервала, признака, диапазона
        # ===========================================
        dbc.Col([
            html.H2("Панель управления", style={"marginBottom": "1rem", "color": "white"}),

            # 1) Выбор «концептуальной» установки
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

            # 2) Интервал интерполяции (мс)
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

            # 3) Выбор признака
            html.Div([
                dbc.Label("Выберите признак", style={"color": "white"}),
                dcc.Dropdown(
                    id="dropdown-feature",
                    options=[],   # заполняется динамически
                    value=None,
                    clearable=False,
                    style={"backgroundColor": "white", "color": "black"},
                    className="mb-4"
                ),
            ]),

            # 4) Диапазон дат
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

            # 5) Статус соединения / сбора данных
            html.Div(
                id="status-message",
                style={"marginTop": "1rem", "color": "#FFD700", "minHeight": "1.5rem"}
            ),

        ], width=3, style={"padding": "1rem", "backgroundColor": "#1A2138", "height": "100vh"}),

        # ===========================================
        #  Правая часть: два графика, метрика, инфо о записях, таблицы out и metrics-файлов
        # ===========================================
        dbc.Col([
            # —————————————————————————————————————————
            # График 1: «сырые» данные (raw_port)
            # —————————————————————————————————————————
            html.H4("Сырые данные", style={"color": "white"}),
            dcc.Graph(
                id="line-chart-raw",
                style={"height": "25vh"}
            ),

            html.Hr(style={"borderColor": "#444"}),

            # —————————————————————————————————————————
            # График 2: «истинные» данные (filled_port)
            # —————————————————————————————————————————
            html.H4("Данные без пропусков", style={"color": "white", "marginTop": "1rem"}),
            dcc.Graph(
                id="line-chart-filled",
                style={"height": "25vh"}
            ),

            html.Hr(style={"borderColor": "#444", "marginTop": "1rem"}),

            # —————————————————————————————————————————
            # График 2: «заполненные» данные (filled_port)
            # —————————————————————————————————————————
            html.H4("Заполнение пропусков", style={"color": "white", "marginTop": "1rem"}),
            dcc.Graph(
                id="line-out-long",
                style={"height": "25vh"}
            ),

            html.Hr(style={"borderColor": "#444", "marginTop": "1rem"}),

            # —————————————————————————————————————————
            # Блок «Метрика модели» (последняя строчка из metrics)
            # —————————————————————————————————————————
            html.H4("Метрика работы модели", style={"color": "white", "marginTop": "1rem"}),
            html.Div(id="metrics-info", style={"color": "white", "marginBottom": "1rem"}),

            # —————————————————————————————————————————
            # Информация о «записях» заполненных данных
            # —————————————————————————————————————————
            html.H4("Информация о записях", style={"color": "white", "marginTop": "1rem"}),
            html.Div(id="data-info", style={"color": "white", "marginBottom": "1rem"}),

            # —————————————————————————————————————————
            # Таблица «out»-файла (теперь из Business/data_out_<port>.csv)
            # —————————————————————————————————————————
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

            # —————————————————————————————————————————
            # Таблица «metrics»-файла (вся история метрик из Business)
            # —————————————————————————————————————————
            html.H4("Таблица metrics (история метрик)", style={"color": "white", "marginTop": "1rem"}),
            dash_table.DataTable(
                id="metrics-file-table",
                columns=[],  # будем заполнять динамически
                data=[],
                page_size=10,
                style_header={"backgroundColor": "#1A2138", "color": "white"},
                style_cell={"backgroundColor": "#202946", "color": "white", "textAlign": "left"},
                style_table={"overflowX": "auto"},
            ),

            # —————————————————————————————————————————
            # Интервал опроса (каждые 2000 мс)
            # —————————————————————————————————————————
            dcc.Interval(
                id="interval-update",
                interval=2000,
                n_intervals=0
            ),
        ], width=9, style={"padding": "1rem"}),

    ])
])


# ----------------------
#  Callback 1: Обновляем список признаков (dropdown-feature)
# ----------------------

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


# ----------------------
#  Callback 2: «Применить интервал» для всех портов
# ----------------------

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
#  Callback 3: Строим два графика, метрику, таблицу «out» и таблицу «metrics»
# ----------------------

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
    raw_port, filled_port = INSTALLATIONS[inst]

    raw_path = os.path.join(RECIEVER_DIR, f"data_port_{raw_port}_long.csv")
    input_path = os.path.join(RECIEVER_DIR, f"data_port_{raw_port}.csv")
    filled_path = os.path.join(RECIEVER_DIR, f"data_port_{filled_port}_long.csv")
    out_path_long = os.path.join(BUSINESS_DIR, f"data_out_{raw_port}_long.csv")

    out_path = os.path.join(BUSINESS_DIR, f"data_out_{filled_port}.csv")

    # Путь к metrics-файлу (в папке Business)
    metrics_path = os.path.join(BUSINESS_DIR, f"data_metrics_{filled_port}.csv")

    # 1) Если нет «длинного» raw – рисуем «пусто»
    if not os.path.exists(raw_path):
        return {}, {}, {}, "", "", "", [], [], []

    # 2) Читаем raw_long
    try:
        df_long = pd.read_csv(raw_path)
    except:
        return {}, {}, {}, "", "", "Ошибка при чтении CSV", [], [], []


    # Если все DateTime пусты → «Потеря соединения»
    if df_long["DateTime"].isnull().all():
        return {}, {}, {}, "", "", "Потеря соединения с установкой", [], [], []

    # 3) Фильтруем по дате для сырых
    dff_raw = df_long.copy()
    if start_date:
        dff_raw = dff_raw[dff_raw["DateTime"] >= start_date]
    if end_date:
        dff_raw = dff_raw[dff_raw["DateTime"] <= end_date]

    # 4) Если feature не в колонках или dff_raw.empty → «Нет данных»
    if not feature or feature not in dff_raw.columns or dff_raw.empty:
        return {}, {}, {}, "", "", "Нет данных для выбранного признака/диапазона", [], [], []

    # 5) Строим график 1: «сырые» данные
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

    # 6) Поток «заполненные» данные (filled_long) не меняется – он нужен для графика 2
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

    # 6) Поток «заполненные» данные (filled_long) не меняется – он нужен для графика 2
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
    out_table_data = []

    if os.path.exists(filled_path):
        try:
            df_filled = pd.read_csv(filled_path)
            dff_filled = df_filled.copy()
            if start_date:
                dff_filled = dff_filled[dff_filled["DateTime"] >= start_date]
            if end_date:
                dff_filled = dff_filled[dff_filled["DateTime"] <= end_date]

            # Определяем y_filled
            if feature in dff_filled.columns:
                used_col = feature
                y_filled = dff_filled[feature]
            else:
                candidates = [c for c in dff_filled.columns if c != "DateTime"]
                if candidates:
                    used_col = candidates[0]
                    y_filled = dff_filled[used_col]
                else:
                    used_col = None
                    y_filled = []

            # График 2: «заполненные» данные
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

                # Информация о записях (filled_long)
                count = len(dff_filled)
                min_date = dff_filled["DateTime"].min()
                max_date = dff_filled["DateTime"].max()
                data_info = (
                    f"Количество записей: {count}. "
                    f"Первая дата: {min_date}. "
                    f"Последняя дата: {max_date}."
                )
            else:
                data_info = "Нет обработанных данных"
        except:
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
            data_info = ""
    else:
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
        data_info = ""

    if os.path.exists(out_path_long):
        try:
            df_out_long = pd.read_csv(out_path_long)
            dff_out_long = df_out_long.copy()
            if start_date:
                dff_out_long = dff_out_long[dff_out_long["DateTime"] >= start_date]
            if end_date:
                dff_out_long = dff_out_long[dff_out_long["DateTime"] <= end_date]

            # Определяем y_filled
            if feature in dff_out_long.columns:
                used_col = feature
                y_filled = dff_out_long[feature]
            else:
                candidates = [c for c in dff_out_long.columns if c != "DateTime"]
                if candidates:
                    used_col = candidates[0]
                    y_filled = dff_out_long[used_col]
                else:
                    used_col = None
                    y_filled = []

            # График 2: «заполненные» данные
            if not dff_out_long.empty and used_col:
                fig_out_long = {
                    "data": [{
                        "x": dff_out_long["DateTime"],
                        "y": y_filled,
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

                # Информация о записях (filled_long)
                count = len(dff_out_long)
                min_date = dff_out_long["DateTime"].min()
                max_date = dff_out_long["DateTime"].max()
                data_info = (
                    f"Количество записей: {count}. "
                    f"Первая дата: {min_date}. "
                    f"Последняя дата: {max_date}."
                )
            else:
                data_info = "Нет обработанных данных"
        except:
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
    else:
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

    # 7) Таблица out: читаем из Business/data_out_<filled_port>.csv
    if os.path.exists(out_path):
        try:
            global df_out, df_input

            if (os.path.getmtime(out_path) >= os.path.getmtime(input_path)) or (df_out is None) or (df_input is None):
                df_out = pd.read_csv(out_path)
                df_input = pd.read_csv(input_path)

            # Фильтруем по дате, если нужно
            dff_out = df_out.copy()
            dff_input = df_input.copy()

            if start_date:
                dff_out = dff_out[dff_out["DateTime"] >= start_date]
                dff_input = dff_input[dff_input["DateTime"] >= start_date]
            if end_date:
                dff_out = dff_out[dff_out["DateTime"] <= end_date]
                dff_input = dff_input[dff_input["DateTime"] <= end_date]

            # Заполняем out_table_data только двумя колонками: DateTime и значение признака
            out_table_data = []
            if feature in dff_out.columns:
                for index, row in dff_out.iterrows():
                    out_table_data.append({
                        "DateTime": row["DateTime"],
                        "input": dff_input.iloc[index][feature],
                        "value": row[feature]
                    })
            else:
                # Если вдруг в out CSV нет выбранного признака,
                # то возьмём первый столбец после DateTime
                cols_out = [c for c in dff_out.columns if c != "DateTime"]
                if cols_out:
                    col0 = cols_out[0]
                    for index, row in dff_out.iterrows():
                        out_table_data.append({
                            "DateTime": row["DateTime"],
                            "input": dff_input.iloc[index][col0],
                            "value": row[col0]
                        })
        except:
            out_table_data = []
    else:
        out_table_data = []

    # 8) Читаем всю историю метрик из Business/data_metrics_<filled_port>.csv
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
    else:
        metrics_file_columns = []
        metrics_file_data = []

    # 9) Текущая метрика: последняя строка из Business/data_metrics_<filled_port>.csv
    metrics_info = ""
    if os.path.exists(metrics_path):
        try:
            dfm_full = pd.read_csv(metrics_path)
            if not dfm_full.empty:
                last = dfm_full.iloc[-1]
                parts = []
                for col in dfm_full.columns:
                    parts.append(f"{col} = {last[col]}")
                metrics_info = ", ".join(parts)
        except:
            metrics_info = ""

    # 10) Статус (оставляем пустым, если всё успешно)
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


# ----------------------
#  Запуск приложения
# ----------------------

if __name__ == "__main__":
    print("Запуск Dash-GUI (Polling-CSV)")
    host=os.getenv("WEBSOCKET_HOST", socket.gethostbyname(socket.gethostname()))
    app.run(debug=True, host=host, port=8099)
