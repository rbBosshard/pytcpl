## pytcpl: A Simplified Python version of  [tcpl: An R package for processing high-throughput chemical screening data](https://github.com/USEPA/CompTox-ToxCast-tcpl)

Welcome to the GitHub repository for the pytcpl package.

The pytcpl package provides a set of tools for processing and modeling high-throughput and high-content chemical screening data of the US EPA ToxCast program. 


### Use conda environment (optional)
- `conda create --name pytcpl`
- `conda activate pytcpl`
- `conda install pip`
- Install packages with pip from requirements.txt:
  - `pip install -r requirements.txt`
- Generate requirements.txt file with pip (only for development):
  - `pipreqs --force`
  - Note: replace problem packages with `sqlalchemy` and `mysql-connector-python`

### Main pipeline script:
- `pytcpl/pipeline.py`

### Config file for pipeline and database login
- `pytcpl/config/config.yaml`

Note: pipeline works only for single assay at once. Loop pipeline for multiple assays.


## Maybe for later:

### Create docs (no docs and not working yet)
- `sphinx-apidoc -o ../docs .`
- `cd ../docs && make clean && make html && cd ../pytcpl`

### Build python package (not working)
- `python -m build`


