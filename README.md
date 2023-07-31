### pytcpl: A lite Python version of [tcpl: An R package for processing high-throughput chemical screening data](https://github.com/USEPA/CompTox-ToxCast-tcpl)

Welcome to the GitHub repository for the pytcpl package.

Pytcpl is a streamlined Python package that incorporates the ***mc4*** and ***mc5*** levels of
[tcpl](https://github.com/USEPA/CompTox-ToxCast-tcpl), 
providing concentration-response curve fitting functionality akin to [tcplfit2](https://github.com/USEPA/CompTox-ToxCast-tcplFit2).
It utilizes the [Invitrodb version 3.5 release](https://cfpub.epa.gov/si/si_public_record_Report.cfm?dirEntryId=355484&Lab=CCTE)
as its backend database.

#### Use conda environment (optional)
- `conda create --name pytcpl`
- `conda activate pytcpl`
- `conda install pip`

#### Install dependencies
  - `pip install -r requirements.txt`

#### Navigate into python package
- `cd pytcpl`

#### Run main pipeline script:
- `python pytcpl/pipeline.py`
- uses config file: `pytcpl/config/config.yaml` (contains database login)
- Note: pipeline works only for single assay at once! 
- Todo: Input a list of assay ids (aeid) to run multiple assays.

#### Visualize curve-fits per assay/chemical pair
- `streamlit run pytcpl/app.py --server.address="localhost"`
- a web app should open in browser
- works of course only for assays that you already run through pipeline
- [Streamlit website](https://streamlit.io/)

#### Run ML for single assay id
- jupyter notebook: `ml.ipynb`
- uses config file: `pytcpl/config/config_ml.yaml`


### This part is only relevant for development!
#### Generate requirements.txt file with pip:
  - `pipreqs --encoding=utf8 --force`
  - Note: remove duplicates and replace problem packages manually
    - `sqlalchemy`
    - `mysql-connector-python`

#### View profiling results (if profiling was activated)
- `pip install snakeviz`
- `snakeviz pytcpl/profile/pipeline.prof`
- call stack with performance details should open in browser
- [SnakeViz website](https://jiffyclub.github.io/snakeviz/)

#### Not working or maybe for later:
Create docs (there are no docstrings at all yet)
- `sphinx-apidoc -o ../docs .`
- `cd ../docs && make clean && make html && cd ../pytcpl`

Build python package
- `python -m build`


