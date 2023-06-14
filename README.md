### pytcpl: A Simplified Python version of  [tcpl: An R package for processing high-throughput chemical screening data](https://github.com/USEPA/CompTox-ToxCast-tcpl)

Welcome to the GitHub repository for the pytcpl package.

The pytcpl package provides a set of tools for processing and modeling high-throughput and high-content chemical screening data of the US EPA ToxCast program. 


#### Use conda environment (optional)
- `conda create --name pytcpl`
- `conda activate pytcpl`
- `conda install pip`

#### Install dependencies
- Install packages with pip from requirements.txt:
  - `pip install -r requirements.txt`

#### Navigate into python package
`cd pytcpl`

#### Main pipeline script:
- `python pytcpl/pipeline.py`

#### Config file for pipeline and database login
- `pytcpl/config/config.yaml`

Note: pipeline works only for single assay at once. Loop pipeline for multiple assays.

#### Visualize curve-fits per assay/chemical pair
- `streamlit run pytcpl/app.py --server.address="localhost"`
- works of course only for assays that you already run through pipeline


<br/><br/>
### Only relevant for development!
#### Generate requirements.txt file with pip (only for development):
  - `pipreqs --force`
  - Note: replace problem packages with `sqlalchemy` and `mysql-connector-python`

#### View profiling results if profile config was set to true
- `pip install snakeviz`
- `snakeviz pytcpl/profile/pipeline.prof`
#### Not working or maybe for later:
Create docs (ther are no docstrings yet)
- `sphinx-apidoc -o ../docs .`
- `cd ../docs && make clean && make html && cd ../pytcpl`

Build python package
- `python -m build`


