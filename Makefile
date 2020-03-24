.PHONY : all tests docs lint_tests code_tests clean install

all : tests docs

tests : code_tests lint_tests

install :
	pip install -r requirements.txt

lint_tests :
	pylint polymath

code_tests :
	py.test --cov polymath --cov-fail-under=100 --cov-report=term-missing --cov-report=html --verbose --durations=5 -s

docs :
	sphinx-build -b doctest docs build
	sphinx-build -nWT docs build

clean :
	rm -rf build/

