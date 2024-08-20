install-deps:
	pip install .

create-env:
	conda env create -f environment.yml

remove-env:
	conda remove -n pytorch --all