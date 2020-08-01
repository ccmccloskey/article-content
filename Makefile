chart:
	python issues/${issue_no}/main.py --chart_id ${chart_id}

test:
	pytest -s --cov-report term-missing --cov=charting/ tests/

check:
	isort

format:
	black .