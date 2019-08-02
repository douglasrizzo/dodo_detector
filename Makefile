tests:
	pip install .[tf-cpu,testing]

ifneq '$USER' 'travis'
	nosetests -s --cover-package=dodo_detector --processes=$(nproc)
else
	nosetests -s --cover-package=dodo_detector
endif

docs:
	pip install .[docs]
	sphinx-build sphinx docs -b html