.PHONY: test

test:
	. bin/activate; pytest --doctest-modules bqme test
