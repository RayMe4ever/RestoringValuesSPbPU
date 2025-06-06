import os
import glob
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

# ===================================================================
# 1. ПАРАМЕТРЫ / УТИЛИТЫ
# ===================================================================

UPDATE_INTERVAL_MS = 2000  # миллисекунды для обновления дашборда
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Reciever")
BUSINESS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Business")

CSV_PATTERN_INPUT = os.path.join(DATA_DIR, "data_port_*.csv")

def find_csv_files():
    """
    Возвращает отсортированный список всех файлов data_port_*.csv в папке Reciever.
    """
    return sorted(glob.glob(CSV_PATTERN_INPUT))


def read_last_n_rows(csv_path: str, n: int = 50) -> pd.DataFrame:
    """
    Читает CSV, пытается распарсить возможный столбец datetime (если есть по типу или по имени),
    сортирует по нему, конвертирует остальные колонки в числовые (float),
    и возвращает последние n строк.
    Если чтение/парсинг неудачный — возвращает пустой DataFrame.
    """
    try:
        df = pd.read_csv(csv_path)

        # Найдём потенциальные datetime-столбцы по типу или ключевым словам
        datetime_cols = [
            c for c in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[c]) 
            or "date" in c.lower() 
            or "time" in c.lower()
        ]
        for col in datetime_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        if datetime_cols:
            df = df.sort_values(datetime_cols[0])

        # Принудительно приводим другие колонки к числу
        for col in df.columns:
            if col in datetime_cols:
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.tail(n)
    except Exception:
        return pd.DataFrame()


def compute_metrics(df: pd.DataFrame, selected_column: str = None) -> dict:
    """
    Вычисляет метрики для DataFrame (после фильтрации по дате):
      - total_rows       : общее число строк
      - avg_selected     : среднее по selected_column (если указан и есть)
      - min_date, max_date: минимальная и максимальная даты из datetime-столбца (если есть)
    """
    metrics = {"total_rows": None, "avg_selected": None, "min_date": None, "max_date": None}
    if df.empty:
        return metrics

    metrics["total_rows"] = len(df)

    if selected_column and selected_column in df.columns and pd.api.types.is_numeric_dtype(df[selected_column]):
        metrics["avg_selected"] = round(df[selected_column].mean(skipna=True), 2)

    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if datetime_cols:
        col = datetime_cols[0]
        s = df[col].dropna()
        if not s.empty:
            metrics["min_date"] = s.min().date()
            metrics["max_date"] = s.max().date()

    return metrics


# ===================================================================
# 2. СОЗДАЕМ DASH-APPLICATION С ТЕМОЙ DARKLY
# ===================================================================

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "CSV Dashboard"


# ===================================================================
# 3. LAYOUT: Sidebar + Основной контент
# ===================================================================

# 3.1. Sidebar с выбором CSV, диапазоном дат и признаком
sidebar = html.Div(
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "280px",
        "padding": "1rem",
        "backgroundColor": "#1f2630",
    },
    children=[
        html.Div(
            [
                html.I(className="bi bi-folder2-open",
                       style={"fontSize": "1.5rem", "color": "white", "marginRight": "0.5rem"}),
                html.Span("Выберите CSV", style={"fontSize": "1.3rem", "fontWeight": "bold", "color": "white"}),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "1rem"},
        ),
        dcc.Dropdown(
            id="csv-dropdown",
            options=[{"label": os.path.basename(p), "value": p} for p in find_csv_files()],
            value=find_csv_files()[0] if find_csv_files() else None,
            clearable=False,
            style={"width": "100%", "color": "#000000"},
        ),

        html.Hr(style={"borderColor": "#ffffff33", "marginTop": "1.5rem", "marginBottom": "1.5rem"}),

        html.Div(
            [
                html.I(className="bi bi-calendar-range",
                       style={"fontSize": "1.5rem", "color": "white", "marginRight": "0.5rem"}),
                html.Span("Диапазон дат", style={"fontSize": "1.3rem", "fontWeight": "bold", "color": "white"}),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "0.5rem"},
        ),
        dcc.DatePickerRange(
            id="date-picker-range",
            display_format="YYYY-MM-DD",
            min_date_allowed=None,
            max_date_allowed=None,
            start_date=None,
            end_date=None,
            style={"width": "100%"},
        ),

        html.Hr(style={"borderColor": "#ffffff33", "marginTop": "1.5rem", "marginBottom": "1.5rem"}),

        html.Div(
            [
                html.I(className="bi bi-bar-chart-line",
                       style={"fontSize": "1.5rem", "color": "white", "marginRight": "0.5rem"}),
                html.Span("Выберите признак", style={"fontSize": "1.3rem", "fontWeight": "bold", "color": "white"}),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "1rem"},
        ),
        dcc.Dropdown(
            id="column-dropdown",
            options=[],
            value=None,
            clearable=False,
            style={"width": "100%", "color": "#000000"},
        ),
    ],
)

# 3.2. Основной контент (справа от sidebar)
content = html.Div(
    style={
        "marginLeft": "280px",
        "padding": "1rem",
        "backgroundColor": "#2d3038",
        "minHeight": "100vh",
    },
    children=[
        # Карточки с метриками
        html.Div(id="metrics-cards", style={"marginBottom": "20px"}),

        # Блок с графиком и таблицей
        html.Div(
            [
                # Левый блок: график
                html.Div(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                html.H6("График признака", style={"color": "white", "margin": 0}),
                                style={
                                    "backgroundColor": "#1f2630",
                                    "borderBottom": "1px solid #ffffff20",
                                    "padding": "0.5rem 1rem"
                                },
                            ),
                            dbc.CardBody(
                                dcc.Graph(id="line-chart", figure={}),
                                style={"backgroundColor": "#2d3038", "padding": "0"},
                            ),
                        ],
                        color="dark",
                        outline=False,
                        style={"borderRadius": "0.5rem", "height": "100%"},
                    ),
                    style={"width": "65%", "display": "inline-block", "verticalAlign": "top"},
                ),

                # Правый блок: таблица
                html.Div(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                html.H6("Таблица данных", style={"color": "white", "margin": 0}),
                                style={
                                    "backgroundColor": "#1f2630",
                                    "borderBottom": "1px solid #ffffff20",
                                    "padding": "0.5rem 1rem"
                                },
                            ),
                            dbc.CardBody(
                                dash_table.DataTable(
                                    id="data-table",
                                    data=[],
                                    columns=[],
                                    page_size=10,
                                    filter_action="native",
                                    sort_action="native",
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "backgroundColor": "#2d3038",
                                        "color": "white",
                                        "minWidth": "80px",
                                        "whiteSpace": "nowrap",
                                        "overflow": "hidden",
                                        "textOverflow": "ellipsis",
                                        "fontSize": "0.85rem",
                                        "padding": "6px",
                                    },
                                    style_header={
                                        "backgroundColor": "#1f2630",
                                        "fontWeight": "bold",
                                        "color": "white",
                                        "fontSize": "0.9rem",
                                    },
                                ),
                                style={"backgroundColor": "#2d3038", "padding": "1rem"},
                            ),
                        ],
                        color="dark",
                        outline=False,
                        style={"borderRadius": "0.5rem", "height": "100%"},
                    ),
                    style={"width": "33%", "display": "inline-block", "marginLeft": "2%", "verticalAlign": "top"},
                ),
            ],
            className="g-4",
        ),

        # Область для текста ошибок/подсказок
        html.Div(id="status-text", style={"color": "white", "marginTop": "20px"}),

        # Таймер для «живого» обновления
        dcc.Interval(id="update-interval", interval=UPDATE_INTERVAL_MS, n_intervals=0),
    ],
)

app.layout = html.Div([sidebar, content])


# ===================================================================
# 4. CALLBACK: при смене CSV заполняем список признаков и диапазон дат
# ===================================================================

@app.callback(
    Output("date-picker-range", "min_date_allowed"),
    Output("date-picker-range", "max_date_allowed"),
    Output("date-picker-range", "start_date"),
    Output("date-picker-range", "end_date"),
    Output("column-dropdown", "options"),
    Output("column-dropdown", "value"),
    Input("csv-dropdown", "value")
)
def update_filters_on_csv_change(selected_csv):
    """
    При выборе нового CSV:
    1) Читаем последние 50 строк.
    2) Определяем первый datetime-столбец (если есть) и заполняем DatePickerRange.
    3) Находим все числовые колонки → возвращаем их как options.
    4) Автоматически выбираем первый признак.
    """
    if not selected_csv or not os.path.exists(selected_csv):
        return None, None, None, None, [], None

    df = read_last_n_rows(selected_csv, n=50)
    if df.empty:
        return None, None, None, None, [], None

    # Найдём datetime-столбцы
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if datetime_cols:
        col = datetime_cols[0]
        min_d = df[col].min().date()
        max_d = df[col].max().date()
        # по умолчанию ставим start=end=min, чтобы показать весь день
        start_d = min_d
        end_d = max_d
    else:
        min_d = max_d = start_d = end_d = None

    # Числовые колонки
    numeric_cols = [
        {"label": c, "value": c}
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    first_val = numeric_cols[0]["value"] if numeric_cols else None

    return min_d, max_d, start_d, end_d, numeric_cols, first_val


# ===================================================================
# 5. CALLBACK: обновление метрик, графика и таблицы каждые UPDATE_INTERVAL_MS
# ===================================================================

@app.callback(
    Output("metrics-cards", "children"),
    Output("line-chart", "figure"),
    Output("data-table", "data"),
    Output("data-table", "columns"),
    Output("status-text", "children"),
    Input("update-interval", "n_intervals"),
    State("csv-dropdown", "value"),
    State("date-picker-range", "start_date"),
    State("date-picker-range", "end_date"),
    State("column-dropdown", "value")
)
def update_dashboard(n_intervals, selected_csv, start_date, end_date, selected_column):
    """
    Каждые UPDATE_INTERVAL_MS:
    - Читает последние 50 строк из выбранного CSV.
    - Фильтрует по дате:
        • Если start_date == end_date, показывает все строки за эту дату.
        • Иначе показывает диапазон [start_date, end_date].
    - Вычисляет метрики (количество строк, среднее по выбранному признаку, первая/последняя даты).
    - Строит график точек с линиями между непропущенными значениями.
    - Заполняет таблицу всеми колонками.
    """
    if not selected_csv or not os.path.exists(selected_csv):
        return [], go.Figure(), [], [], "Ошибка: CSV-файл не найден."

    df = read_last_n_rows(selected_csv, n=50)
    if df.empty:
        return [], go.Figure(), [], [], "Ошибка: не удалось прочитать CSV или он пустой."

    # Фильтрация по дате, если есть datetime-столбец
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if datetime_cols and start_date and end_date:
        col = datetime_cols[0]
        s = df[col]
        sd = pd.to_datetime(start_date)
        ed = pd.to_datetime(end_date)

        if sd.date() == ed.date():
            # если даты совпадают, выбираем все строки, где date == sd.date()
            mask = s.dt.date == sd.date()
        else:
            # диапазон от sd 00:00 до ed 23:59:59
            mask = (s >= sd) & (s <= (ed + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)))
        dff = df.loc[mask].copy()
    else:
        dff = df.copy()

    if dff.empty:
        return [], go.Figure(), [], [], "После фильтрации по дате данных нет."

    # Метрики
    metrics = compute_metrics(dff, selected_column)
    cards = []

    if metrics["total_rows"] is not None:
        cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6("Всего записей", className="card-title", style={"color": "white"}),
                            html.H2(f"{metrics['total_rows']:,}", style={"color": "#00d9ff", "fontWeight": "bold"}),
                            html.Div("после фильтрации", style={"color": "#bbbbbb", "fontSize": "0.85rem"}),
                        ]
                    ),
                    color="dark",
                    outline=False,
                    style={"borderRadius": "0.5rem", "backgroundColor": "#1f2630"},
                ),
                xs=12, sm=6, md=3, lg=3
            )
        )

    if metrics["avg_selected"] is not None:
        cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6("Среднее (выбранный признак)", className="card-title", style={"color": "white"}),
                            html.H2(f"{metrics['avg_selected']}", style={"color": "#ffbf00", "fontWeight": "bold"}),
                            html.Div("", style={"color": "#bbbbbb", "fontSize": "0.85rem"}),
                        ]
                    ),
                    color="dark",
                    outline=False,
                    style={"borderRadius": "0.5rem", "backgroundColor": "#1f2630"},
                ),
                xs=12, sm=6, md=3, lg=3
            )
        )

    if metrics["min_date"] is not None:
        cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6("Первая дата", className="card-title", style={"color": "white"}),
                            html.H2(f"{metrics['min_date']}", style={"color": "#50db50", "fontWeight": "bold"}),
                            html.Div("начало периода", style={"color": "#bbbbbb", "fontSize": "0.85rem"}),
                        ]
                    ),
                    color="dark",
                    outline=False,
                    style={"borderRadius": "0.5rem", "backgroundColor": "#1f2630"},
                ),
                xs=12, sm=6, md=3, lg=3
            )
        )

    if metrics["max_date"] is not None:
        cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6("Последняя дата", className="card-title", style={"color": "white"}),
                            html.H2(f"{metrics['max_date']}", style={"color": "#ff6384", "fontWeight": "bold"}),
                            html.Div("конец периода", style={"color": "#bbbbbb", "fontSize": "0.85rem"}),
                        ]
                    ),
                    color="dark",
                    outline=False,
                    style={"borderRadius": "0.5rem", "backgroundColor": "#1f2630"},
                ),
                xs=12, sm=6, md=3, lg=3
            )
        )

    cards_row = dbc.Row(cards, className="g-4", style={"marginBottom": "20px"})

    # График точек с линиями между непропущенными значениями
    if not selected_column or selected_column not in dff.columns:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Выберите корректный признак для графика",
            plot_bgcolor="#2d3038",
            paper_bgcolor="#2d3038",
            font={"color": "white"},
            margin={"t": 40, "b": 40, "l": 40, "r": 20},
        )
        fig = empty_fig
    else:
        x_vals, y_vals = [], []
        for i, val in enumerate(dff[selected_column]):
            if pd.isna(val):
                x_vals.append(None)
                y_vals.append(None)
            else:
                x_vals.append(i)
                y_vals.append(val)

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    name=selected_column,
                    connectgaps=False  # линии не соединяют пропущенные точки
                )
            ],
            layout=go.Layout(
                title=f"{selected_column} over Index",
                xaxis={"title": "Index", "color": "white"},
                yaxis={"title": selected_column, "color": "white"},
                plot_bgcolor="#2d3038",
                paper_bgcolor="#2d3038",
                font={"color": "white"},
                margin={"t": 40, "b": 40, "l": 40, "r": 20},
                uirevision="constant"
            )
        )

    # Таблица (все колонки)
    table_data = dff.to_dict("records")
    table_columns = [{"name": col, "id": col} for col in dff.columns]

    return cards_row, fig, table_data, table_columns, ""


# ===================================================================
# 6. ЗАПУСК СЕРВЕРА
# ===================================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
