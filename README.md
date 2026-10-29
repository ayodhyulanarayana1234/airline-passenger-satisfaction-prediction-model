# Airline Passenger Satisfaction Prediction

This project explores airline passenger satisfaction and compares several
classification models.

## Dataset

The dataset is not included in this repository. Place
`Airline Passenger Satisfaction.csv` in the same folder as the Python script
before running it. The script loads the CSV from its own folder.

## Setup

Install Python 3 and the packages listed in `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Then run the analysis:

```powershell
python ".\Predictive Project.py"
```

The script prints data summaries, trains and compares classification models,
and displays exploratory-analysis and model-result plots.
