VENV := .venv

# Windows venvs put the interpreter under Scripts/, not bin/, and Windows
# Python installers usually only provide `python`, not `python3`.
ifeq ($(OS),Windows_NT)
    PY        := $(VENV)/Scripts/python.exe
    STREAMLIT := $(VENV)/Scripts/streamlit.exe
    PYTHON    := python
else
    PY        := $(VENV)/bin/python
    STREAMLIT := $(VENV)/bin/streamlit
    PYTHON    := python3
endif

.PHONY: setup test app demo lint clean

setup:            ## create a venv and install dependencies
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip -q
	$(PY) -m pip install -r requirements.txt -q
	@echo "done. next:  make test"

test:             ## run the offline contract tests
	cd src && ../$(PY) -m pytest -q

app:              ## launch the Streamlit UI (offline by default)
	cd src && ../$(STREAMLIT) run itinerary_app.py

demo:             ## run one goal end-to-end in the terminal and print the trace
	cd src && ../$(PY) run_once.py "Plan a one-day layover in Tokyo on a 5000 yen budget."

clean:
	rm -rf $(VENV) .pytest_cache src/__pycache__ src/tests/__pycache__
