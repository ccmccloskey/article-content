import calendar
import datetime as dt
import random
from collections import OrderedDict
from enum import Enum

import numpy as np
import plotly
import plotly.graph_objects as go


class NeutralTimeSeriesColorPalette(Enum):
    white = "white"
    grey = "rgb(217,217,217)"

    @classmethod
    def to_value_list(cls):
        return [color.value for color in cls]

    @classmethod
    def to_dict(cls, color_name_order=None):
        colors = {color.name: color.value for color in cls}
        if color_name_order:
            colors = cls.to_ordered_dict(colors, color_name_order)
        return colors

    @staticmethod
    def to_ordered_dict(colors, color_name_order):
        return OrderedDict(
            (color_name, colors[color_name]) for color_name in color_name_order
        )


class DefaultSeriesColorPalette(Enum):
    first = "rgb(0,63,92)"
    second = "rgb(88,80,141)"
    third = "rgb(188,80,144)"
    fourth = "rgb(255,99,97)"
    fifth = "rgb(255,166,0)"

    @classmethod
    def to_value_list(cls):
        return [color.value for color in cls]

    @classmethod
    def to_dict(cls, color_name_order=None):
        colors = {color.name: color.value for color in cls}
        if color_name_order:
            colors = cls.to_ordered_dict(colors, color_name_order)
        return colors

    @staticmethod
    def to_ordered_dict(colors, color_name_order):
        return OrderedDict(
            (color_name, colors[color_name]) for color_name in color_name_order
        )


class SentimentColorPalette(Enum):
    negative = "rgb(239,59,44)"
    neutral = "rgb(115,115,115)"
    positive = "rgb(65,171,93)"

    @classmethod
    def to_value_list(cls, color_name_order=None):
        colors = {color.name: color.value for color in cls}
        if color_name_order:
            colors = cls.to_ordered_dict(colors, color_name_order)
        return [color for color in colors.values()]

    @classmethod
    def to_dict(cls, color_name_order=None):
        colors = {color.name: color.value for color in cls}
        if color_name_order:
            colors = cls.to_ordered_dict(colors, color_name_order)
        return colors

    @staticmethod
    def to_ordered_dict(colors, color_name_order):
        return OrderedDict(
            (color_name, colors[color_name]) for color_name in color_name_order
        )


class DarkSentimentColorPalette(Enum):
    negative = "rgb(103,0,13)"
    neutral = "rgb(0,0,0)"
    positive = "rgb(0,68,27)"

    @classmethod
    def to_value_list(cls, color_name_order=None):
        colors = {color.name: color.value for color in cls}
        if color_name_order:
            colors = cls.to_ordered_dict(colors, color_name_order)
        return [color for color in colors.values()]

    @classmethod
    def to_dict(cls, color_name_order=None):
        colors = {color.name: color.value for color in cls}
        if color_name_order:
            colors = cls.to_ordered_dict(colors, color_name_order)
        return colors

    @staticmethod
    def to_ordered_dict(colors, color_name_order):
        return OrderedDict(
            (color_name, colors[color_name]) for color_name in color_name_order
        )


to_dark_palette_map = {
    SentimentColorPalette: DarkSentimentColorPalette,
}


class ColorPaletteTransformer:
    def __init__(self, palette, color_name_order=None):
        self.palette = palette
        self.color_dict = palette.to_dict(color_name_order=color_name_order)

    def dim_colors(self, *color_names):
        for color_name in filter(
            lambda color_name: color_name in color_names, self.color_dict.keys()
        ):
            self.color_dict[color_name] = ColorPaletteTransformer.add_opacity_to_color(
                self.color_dict[color_name], 0.1
            )

    def darken_colors(self, *color_names):
        self.dark_palette = to_dark_palette_map.get(self.palette)
        if self.dark_palette:
            for color_name in filter(
                lambda color_name: color_name in color_names, self.color_dict.keys()
            ):
                self.color_dict[color_name] = self.dark_palette.to_dict()[color_name]
        else:
            raise KeyError(
                f"No darkened color palette exists for color palette {self.palette}"
            )

    def create_continuous_scale_for_color(
        self, color_name, n, opacity_step=0.1, is_reversed=True
    ):
        color_value = self.color_dict[color_name]
        opacitys = np.arange(0.1, 1.1, opacity_step)
        if is_reversed:
            opacitys = reversed(opacitys)

        opacitys = list(opacitys)
        if n > len(opacitys):
            opacitys.extend([opacitys[-1]] * (n - len(opacitys)))
        opacitys = opacitys[:n]

        scale = [
            ColorPaletteTransformer.add_opacity_to_color(color_value, o)
            for o in opacitys
        ]

        self.color_dict.update({color_name: scale})

    def create_value_list(self):
        value_list = []
        for color_value in self.color_dict.values():
            if isinstance(color_value, list):
                value_list.extend(color_value)
            else:
                value_list.append(color_value)
        self.value_list = value_list

    @staticmethod
    def add_opacity_to_color(color_value, opacity):
        replaced_color = color_value.replace("rgb(", "rgba(")
        return replaced_color.replace(")", f",{opacity})")

    @staticmethod
    def reset_series_color_index_value_and_get_color(color_palette_list, index):
        return color_palette_list[index % len(color_palette_list)]

    def get_color_value(self, color_name):
        return self.color_dict[color_name]


class BaseHorizontalBarChart:
    def __init__(
        self,
        y,
        color_palette=DefaultSeriesColorPalette.to_value_list(),
        title=None,
        xaxis_title=None,
        yaxis_title=None,
        *data_series,
    ):
        self.data_series = data_series
        self.y = y
        self.xaxis = dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            zeroline=False,
            title=xaxis_title,
            tickfont=dict(family="Roboto", size=16, color="rgb(8,48,107)"),
            titlefont=dict(family="Roboto-Thin", size=30, color="rgb(8,48,107)"),
        )
        self.yaxis = dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            zeroline=False,
            title=yaxis_title,
            tickfont=dict(family="Roboto", size=16, color="rgb(8,48,107)"),
            titlefont=dict(family="Roboto-Thin", size=30, color="rgb(8,48,107)"),
        )
        self.title = dict(text=title, xanchor="center", xref="paper")
        self.titlefont = dict(family="Roboto-Black", size=40, color="rgb(8,48,107)")
        self.showlegend = False
        self.plot_bgcolor = "white"
        self.series_color_palette = color_palette
        self.series_count = len(self.data_series)
        self.annotations = None
        self.barmode = "group"

    def create_figure(self):
        self.fig = go.Figure(
            data=[
                go.Bar(
                    name=s["name"].capitalize(),
                    x=s["x"],
                    y=self.y,
                    width=[0.2] * len(s["x"]),
                    marker_color=ColorPaletteTransformer.reset_series_color_index_value_and_get_color(
                        self.series_color_palette, i
                    ),
                    orientation="h",
                )
                for i, s in enumerate(self.data_series)
            ]
        )

    def update_layout(self, **kwargs):
        set_values = {
            "xaxis": self.xaxis,
            "yaxis": self.yaxis,
            "title": self.title,
            "showlegend": self.showlegend,
            "plot_bgcolor": self.plot_bgcolor,
            "titlefont": self.titlefont,
            "annotations": self.annotations,
            "barmode": self.barmode,
        }
        self.fig.update_layout(dict(set_values, **kwargs))

    def update_xaxis_configuration(self, **params):
        for key, value in params.items():
            self.xaxis.update({key: value})

    def update_yaxis_configuration(self, **params):
        for key, value in params.items():
            self.yaxis.update({key: value})

    @staticmethod
    def boldify_text(text):
        return "<b>" + text + "</b>"

    @staticmethod
    def italicize_text(text):
        return "<i>" + text + "</i>"

    def show_figure(self):
        self.fig.show()

    def save_figure_as_json(self, f):
        self.fig.write_json(f, pretty=True)

    def save_figure_as_image(self, f, fmt="png"):
        self.fig.write_image(f, fmt)

    def save_figure_as_html(self, f, auto_open):
        plotly.offline.plot(self.fig, filename=f, auto_open=auto_open)


class SentimentHorizontalBarChartWithComment(BaseHorizontalBarChart):
    def __init__(self, y, color_palette, title=None, yaxis_title=None, *data_series):
        super().__init__(
            y, color_palette, title, "Sentiment Breakdown", yaxis_title, *data_series
        )
        self.update_xaxis_configuration(
            tickformat=",.0%", hoverformat=",.0%", tickvals=[0, 0.25, 0.50, 0.75, 1],
        )
        self.update_yaxis_configuration(showticklabels=False)
        self.barmode = "stack"
        self.series_color_palette = color_palette
        self.showlegend = False

    def append_comments_to_annotations(self, *comments):

        title_texts = [self.boldify_text(text.upper()) for text in self.y]
        comment_texts = [self.italicize_text(c) for c in comments]

        yref_uplift = 1 / len(title_texts)
        yref_downlift = yref_uplift * 0.25

        yref_value = 0
        self.annotations = []
        for tit, cot in zip(title_texts, comment_texts):
            yref_value += yref_uplift
            self.annotations.append(
                dict(
                    xref="x",
                    x=0.5,
                    yref="paper",
                    y=yref_value,
                    xanchor="center",
                    yanchor="middle",
                    text=tit,
                    font=dict(family="Roboto", size=16, color="rgb(8,48,107)"),
                    showarrow=False,
                )
            )

            self.annotations.append(
                dict(
                    xref="x",
                    x=0.5,
                    yref="paper",
                    y=yref_value - yref_downlift,
                    xanchor="center",
                    yanchor="middle",
                    text=cot,
                    font=dict(family="Roboto Thin", size=20, color="rgb(8,48,107)"),
                    showarrow=False,
                )
            )


class BaseTimeSeriesChart:
    def __init__(
        self,
        dates,
        color_palette=DefaultSeriesColorPalette.to_value_list(),
        title=None,
        xaxis_title=None,
        yaxis_title=None,
        *data_series,
    ):
        self.data_series = data_series
        self.dates = dates
        self.xaxis = dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            title=xaxis_title,
            zeroline=False,
            tickformat="%b",
            hoverformat="%b %d",
            tickfont=dict(family="Roboto", size=16, color="rgb(217,217,217)"),
        )
        self.yaxis = dict(
            showgrid=False,
            showticklabels=True,
            title=yaxis_title,
            tickfont=dict(family="Roboto", size=16, color="rgb(217,217,217)"),
        )
        self.title = dict(text=title, xanchor="center", xref="paper")
        self.titlefont = dict(family="Roboto-Black", size=40, color="rgb(8,48,107)")
        self.series_color_palette = color_palette
        self.series_count = len(self.data_series)
        self.annotations = []
        self.shapes = None
        self.showlegend = False
        self.plot_bgcolor = "white"
        self.unique_year_months = set([(d.year, d.month) for d in self.dates])

    def create_figure(self):
        self.fig = go.Figure()
        for i, s in enumerate(self.data_series):
            color = ColorPaletteTransformer.reset_series_color_index_value_and_get_color(
                self.series_color_palette, i
            )
            self.fig.add_trace(
                go.Scatter(
                    x=self.dates,
                    y=s["y"],
                    mode="lines",
                    line_shape="spline",
                    name=s["name"].capitalize(),
                    line=dict(color=color, width=4),
                )
            )
            self.fig.add_trace(
                go.Scatter(
                    x=[self.dates[0]],
                    y=[s["y"][0]],
                    mode="markers",
                    marker=dict(color=color, size=14),
                    marker_symbol="square",
                    hoverinfo="skip",
                )
            )
            self.fig.add_trace(
                go.Scatter(
                    x=[self.dates[-1]],
                    y=[s["y"][-1]],
                    mode="markers",
                    marker=dict(color=color, size=20),
                    marker_symbol="hexagon",
                    hoverinfo="skip",
                )
            )
            self.annotations.append(
                dict(
                    xref="x",
                    x=self.dates[-1],
                    yref="y",
                    y=s["y"][-1],
                    xanchor="left",
                    xshift=10,
                    yanchor="middle",
                    text=s["name"].capitalize(),
                    font=dict(family="Roboto", size=16, color=color),
                    showarrow=False,
                )
            )

    def update_layout(self, **kwargs):
        set_values = {
            "xaxis": self.xaxis,
            "yaxis": self.yaxis,
            "title": self.title,
            "showlegend": self.showlegend,
            "plot_bgcolor": self.plot_bgcolor,
            "titlefont": self.titlefont,
            "annotations": self.annotations,
            "shapes": self.shapes,
        }
        self.fig.update_layout(dict(set_values, **kwargs))

    def update_xaxis_configuration(self, **params):
        for key, value in params.items():
            self.xaxis.update({key: value})

    def update_yaxis_configuration(self, **params):
        for key, value in params.items():
            self.yaxis.update({key: value})

    @staticmethod
    def boldify_text(text):
        return "<b>" + text + "</b>"

    @staticmethod
    def italicize_text(text):
        return "<i>" + text + "</i>"

    def show_figure(self):
        self.fig.show()

    @staticmethod
    def get_number_of_days_in_year_month(year, month):
        return calendar.monthrange(year, month)[1]

    @staticmethod
    def get_middle_datetime_in_year_month(year, month):
        number_of_days_in_year_month = BaseTimeSeriesChart.get_number_of_days_in_year_month(
            year, month
        )
        first, last = (
            dt.datetime(year, month, 1),
            dt.datetime(year, month, number_of_days_in_year_month),
        )
        return first + ((last - first) / 2)

    def create_date_axis_configuration(self):
        neutral_time_series_color_palette = (
            NeutralTimeSeriesColorPalette.to_value_list()
        )

        xaxis_tickvals = []
        self.shapes = []
        for i, (year, month) in enumerate(self.unique_year_months):
            self.shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=dt.datetime(year, month, 1).date(),
                    y0=0,
                    x1=dt.datetime(year, month + 1, 1).date(),
                    y1=1,
                    fillcolor=ColorPaletteTransformer.reset_series_color_index_value_and_get_color(
                        neutral_time_series_color_palette, i
                    ),
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )
            )
            xaxis_tickvals.append(
                BaseTimeSeriesChart.get_middle_datetime_in_year_month(year, month)
            )

        self.update_xaxis_configuration(tickvals=xaxis_tickvals)

    def add_events_to_timeseries_plot(self, color, *events):
        for event in events:
            self.fig.add_trace(
                go.Scatter(
                    x=[event.date],
                    y=[event.y],
                    mode="markers",
                    marker=dict(color=color, size=16),
                    marker_symbol="hexagon",
                    hoverinfo="skip",
                )
            )
            self.fig.layout.annotations += (
                dict(
                    xref="x",
                    x=event.date,
                    yref="y",
                    y=event.y,
                    xanchor="center",
                    yanchor="bottom",
                    yshift=5,
                    text=self.boldify_text(event.display_text),
                    font=dict(family="Roboto", size=14, color=color),
                    showarrow=False,
                ),
            )

    def save_figure_as_json(self, f):
        self.fig.write_json(f, pretty=True)

    def save_figure_as_image(self, f, fmt="png"):
        self.fig.write_image(f, fmt)


class TimeSeriesEvent:
    def __init__(self, date, y, display_text):
        self.date = date
        self.y = y
        self.display_text = display_text
