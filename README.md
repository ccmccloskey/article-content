## Medium Articles Chart Content

## Setup
You must have pipenv to install dependencies. On a Mac, using Brew, do
```
brew install pipenv
```

Follow equivalent instruction for a Linux and Windows based OS.

Install deps
```
pipenv install --dev
```

## Run Tests
To run tests, do
```
make test
```

## Show a Chart
```
make chart issue_no=<issue_no> chart_id=<chart_id>
```

For example

```
make chart issue_no=i001 chart_id=ts-a
```

Current List of "issue_no-chart_id"'s  available:

| issue_no | chart_id |
|----------|----------|
| i001     | hb-a     |
| i001     | hb-b     |
| i001     | ts-a     |
| i001     | ts-b     |
