from charting.charting import (BaseTimeSeriesChart, SentimentColorPalette,
                               SentimentHorizontalBarChartWithComment)
from charting.utils import GetChart


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
