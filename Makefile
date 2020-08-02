chart:
	pipenv run python issues/${issue_no}/main.py --chart_id ${chart_id}

test:
	pipenv run pytest -s --cov-report term-missing --cov=charting/ tests/

check:
	pipenv run flake8 --exclude issues/*/scripting/
	pipenv run black --check .

format:
	pipenv run isort -y -m 3 -tc
	pipenv run black .
