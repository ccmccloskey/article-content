import json
import time

import pytest

from issues.i001 import main


@pytest.mark.parametrize("issue_no,chart_id,fn", [("i001", "hb-a", main.GetHBA),])
def test_get_chart(issue_no, chart_id, fn):
    """runs use case, asserts json 

    Arguments:
        issue_no {bool} -- [description]
        chart_id {[type]} -- [description]
        fn {function} -- [description]

    Raises:
        e: if assertion fails, prompts dev to inspect chart
           (Y) persists json and html into prod data dirs and removes from test dirs
           (N) raises assertion error
    """
    chart_builder_test = fn(run_mode="test")
    persist_html = chart_builder_test.run()

    try:
        with open(f"issues/{issue_no}/data/{issue_no}-{chart_id}.json") as f:
            expected_json = json.load(f)
        with open(f"tests/data/{issue_no}-{chart_id}.json") as f:
            actual_json = json.load(f)
        assert actual_json == expected_json
        print("SUCCESS")
        chart_builder_test.remove_persisted_data()
    except AssertionError as e:
        persist_html(auto_open=True)
        time.sleep(1)
        fn(run_mode="cli").load_html_in_browser()
        result = input("Persist Changes? [Y/N]")
        if result == "Y":
            print("SUCCESS")
            chart_builder_test.move_persisted_data(switch_run_mode="cli")
        else:
            print("FAIL")
            raise e