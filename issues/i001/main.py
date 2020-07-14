from charting.charting import (BaseTimeSeriesChart, SentimentColorPalette,
                               SentimentHorizontalBarChartWithComment, ColorPaletteTransformer, DarkSentimentColorPalette, TimeSeriesEvent)
from charting.utils import GetChart
import numpy as np
import datetime as dt


class GetHBA(GetChart):
    def __init__(self, run_mode):
        super().__init__("i001", "hb-a", run_mode)

    def build_chart(self):
        self.chart = SentimentHorizontalBarChartWithComment(
            self.data["y"],
            SentimentColorPalette.to_value_list(
                color_name_order=["negative", "neutral", "positive"]
            ),
            None,
            None,
            *self.data["data_series"]
        )
        self.chart.create_figure()
        self.chart.append_comments_to_annotations(*self.data["commments"])
        self.chart.update_xaxis_configuration(
            titlefont=dict(family="Roboto", size=20, color="rgb(8,48,107)")
        )
        self.chart.update_layout(showlegend=False)
        self.chart.show_figure()


class GetHBB(GetChart):
    def __init__(self, run_mode):
        super().__init__("i001", "hb-b", run_mode)
        self.colors = ColorPaletteTransformer(
            SentimentColorPalette, color_name_order=['negative', 'neutral', 'positive'])
        self.colors.darken_colors('negative')
        self.colors.dim_colors('positive', 'neutral')
        self.colors.create_continuous_scale_for_color('negative', 5, 0.2)
        self.colors.create_value_list()

    def build_chart(self):
        self.chart = SentimentHorizontalBarChartWithComment(
            self.data["y"],
            self.colors.value_list,
            None,
            None,
            *self.data["data_series"]
        )
        self.chart.create_figure()
        self.chart.append_comments_to_annotations(*self.data["commments"])
        self.chart.update_xaxis_configuration(
            titlefont=dict(family="Roboto", size=20, color="rgb(8,48,107)")
        )
        self.chart.update_layout(showlegend=False)
        self.chart.show_figure()


class GetTSA(GetChart):
    def __init__(self, run_mode):
            super().__init__("i001", "ts-a", run_mode)

    def build_chart(self):
        self.chart = BaseTimeSeriesChart(
            self.data["dates"],
            [DarkSentimentColorPalette.negative.value],
            None,
            None,
            None,
            *self.data["data_series"],
        )
        self.chart.create_figure()
        self.chart.update_yaxis_configuration(
            tickvals=np.arange(0, 1, 0.01),
            hoverformat=",.1%",
        )
        self.chart.create_date_axis_configuration()
        self.chart.update_layout()
        self.chart.show_figure()


class GetTSB(GetChart):
    def __init__(self, run_mode):
            super().__init__("i001", "ts-b", run_mode)
            self.colors = ColorPaletteTransformer(
                DarkSentimentColorPalette, color_name_order=["negative", "neutral", "positive"])
            self.colors.dim_colors("negative")
            self.colors.create_value_list()

    def build_chart(self):
        self.chart = BaseTimeSeriesChart(
            self.data["dates"],
            [self.colors.value_list[0]],
            None,
            None,
            None,
            *self.data["data_series"],
        )
        self.chart.create_figure()
        self.chart.update_yaxis_configuration(
            tickvals=np.arange(0, 1, 0.01),
            hoverformat=",.1%",
        )
        self.chart.create_date_axis_configuration()
        self.chart.update_layout()
        self.chart.add_events_to_timeseries_plot(
            DarkSentimentColorPalette.negative.value,
            TimeSeriesEvent(dt.datetime(2020, 3, 11), self.data["data_series"][0]["y"][self.data["dates"].index(dt.date(2020, 3, 11))], 'WHO declares the coronavirus outbreak a pandemic'),
            TimeSeriesEvent(dt.datetime(2020, 3, 22), self.data["data_series"][0]["y"][self.data["dates"].index(dt.date(2020, 3, 22))], 'NYC Lockdown Begins'),
            TimeSeriesEvent(dt.datetime(2020, 3, 23), self.data["data_series"][0]["y"][self.data["dates"].index(dt.date(2020, 3, 23))], 'London (UK) Lockdown Begins'))
        self.chart.show_figure()

chart_builder = GetTSB(run_mode="cli")
chart_builder.build_chart()
