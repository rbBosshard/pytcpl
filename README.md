### pytcpl: A lite Python version of  [tcpl: An R package for processing high-throughput chemical screening data](https://github.com/USEPA/CompTox-ToxCast-tcpl)

Welcome to the GitHub repository for the pytcpl package.

The pytcpl package provides a set of tools for processing and 
modeling high-throughput and high-content chemical screening data 
of the US EPA ToxCast program. 

#### Use conda environment (optional)
- `conda create --name pytcpl`
- `conda activate pytcpl`
- `conda install pip`

#### Install dependencies
  - `pip install -r requirements.txt`

#### Navigate into python package
- `cd pytcpl`

#### Config file for pipeline and database login
- `pytcpl/config/config.yaml`
- Note: pipeline works only for single assay at once. 
- Todo: Input a list of assay ids (aeid) to run multiple assays.

#### Run main pipeline script:
- `python pytcpl/pipeline.py`

#### Visualize curve-fits per assay/chemical pair
- `streamlit run pytcpl/app.py --server.address="localhost"`
- a web app should open in browser
- works of course only for assays that you already run through pipeline
- [Streamlit website](https://streamlit.io/)


### This part is only relevant for development!
#### Generate requirements.txt file with pip:
  - `pipreqs --force`
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


