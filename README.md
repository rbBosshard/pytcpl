### Main run script:
`python pipeline.py`

### Config file for database login
`config.yaml`

### Generate a requirements.txt file with pip
`pip freeze > requirements.txt`

### Install packages with pip from requirements.txt:
`pip install -r requirements.txt`

### Create docs
- `sphinx-apidoc -o docs .`
- `cd docs && make html && cd ..`
