python3 setup.py clean
python3 setup.py sdist bdist_wheel
twine upload dist/*