install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest --nbval notebook/EDA.ipynb
	
format:
	isort *.py
	black *.py

run:
	python main.py

lint:
	pylint --disable=R,C main.py

all: install lint