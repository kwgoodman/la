# la Makefile 

srcdir := la/src

help:
	@echo "Available tasks:"
	@echo "help   -->  This help page"
	@echo "build  -->  Build the Cython extension modules"
	@echo "clean  -->  Remove all the build files for a fresh start"
	@echo "test   -->  Run unit tests"
	@echo "info   -->  Info about your la installation"
	@echo "all    -->  clean, build, test, info"
	@echo "sdist  -->  Make source distribution"

all: clean build test info

build:
	rm -rf ${srcdir}/../clabel.so
	python ${srcdir}/setup.py build_ext --inplace
		
test:
	python -c "import la; la.test()"

sdist:
	rm -f MANIFEST
	git status
	python setup.py sdist

info:
	python -c "import la; la.info()"

# Phony targets for cleanup and similar uses

.PHONY: clean
clean:
	rm -rf build
	rm -rf dist
	rm -rf la/cflabel.so
	rm -rf ${srcdir}/build
