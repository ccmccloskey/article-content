import datetime
import os
import webbrowser
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial

import jsonplus as json


class RunMode(Enum):
    test = "test"
    cli = "cli"


class GetChart(ABC):
    def __init__(self, issue_no, chart_id, run_mode: RunMode):
        self.issue_no = issue_no
        self.chart_id = chart_id
        self.run_mode = run_mode
        with open(f"issues/{issue_no}/data/pl-{chart_id}.json") as f:
            self.data = json.loads(f.read())
        with open("charting/persisted_data_config.json") as f:
            self.config = json.loads(f.read())
        self.persist_html = self.config[run_mode]["html"]
        self.persist_json = self.config[run_mode]["json"]
        self.persist_path = os.path.realpath(
            self.replace_issue_no_placeholder_for_path(
                self.config[run_mode]["path"], self.issue_no
            )
        )
        super().__init__()

    @staticmethod
    def replace_issue_no_placeholder_for_path(path, issue_no):
        return f"{issue_no}".join(path.split("{issue_no}"))

    @abstractmethod
    def build_chart(self):
        pass

    @staticmethod
    def get_path_for_json(persist_path, issue_no, chart_id):
        return f"{persist_path}/{issue_no}-{chart_id}.json"

    @staticmethod
    def get_path_for_html(persist_path, issue_no, chart_id):
        return f"{persist_path}/{issue_no}-{chart_id}.html"

    def handle_persisted_data(self):
        if self.persist_json:
            json_path = self.get_path_for_json(
                self.persist_path, self.issue_no, self.chart_id
            )
            self.chart.save_figure_as_json(json_path)
            print(f"persisted into {json_path}")

        if self.persist_html:
            html_path = self.get_path_for_html(
                self.persist_path, self.issue_no, self.chart_id
            )
            persist_html = partial(
                self.chart.save_figure_as_html,
                self.get_path_for_html(self.persist_path, self.issue_no, self.chart_id),
            )
            persist_html(auto_open=False)
            print(f"persisted into {html_path}")

            return persist_html

    def move_persisted_data(self, switch_run_mode: RunMode):
        self.switch_persist_path = os.path.realpath(
            self.replace_issue_no_placeholder_for_path(
                self.config[switch_run_mode]["path"], self.issue_no
            )
        )

        old_json_path = self.get_path_for_json(
            self.persist_path, self.issue_no, self.chart_id
        )
        new_json_path = self.get_path_for_json(
            self.switch_persist_path, self.issue_no, self.chart_id
        )

        os.rename(old_json_path, new_json_path)
        print(f"moved {old_json_path} to {new_json_path}")

        old_html_path = self.get_path_for_html(
            self.persist_path, self.issue_no, self.chart_id
        )
        new_html_path = self.get_path_for_html(
            self.switch_persist_path, self.issue_no, self.chart_id
        )

        os.rename(old_html_path, new_html_path)
        print(f"moved {old_html_path} to {new_html_path}")

    def remove_persisted_data(self):
        json_path = self.get_path_for_json(
            self.persist_path, self.issue_no, self.chart_id
        )
        os.remove(json_path)
        print(f"removed {json_path}")

        html_path = self.get_path_for_html(
            self.persist_path, self.issue_no, self.chart_id
        )
        os.remove(html_path)
        print(f"removed {html_path}")

    def load_html_in_browser(self):
        path = self.get_path_for_html(self.persist_path, self.issue_no, self.chart_id)
        return webbrowser.open("file://" + os.path.realpath(path))

    def run(self):
        self.build_chart()
        return self.handle_persisted_data()
