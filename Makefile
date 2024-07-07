install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest test.py
	#--nbval notebook/EDA.ipynb notebook/MODEL_TRAINING.ipynb

format:
	isort *.py
	black *.py

run:
	python main.py

lint:
	pylint src/pipeline/*.py #src/*.py src/components/*.py

all: install format lint