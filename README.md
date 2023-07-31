## _pytcpl_: A lite Python version of [tcpl: An R package for processing high-throughput chemical screening data](https://github.com/USEPA/CompTox-ToxCast-tcpl)

Welcome to the GitHub repository for the _pytcpl_ package.

_pytcpl_ is a streamlined Python package that incorporates the **mc4** and **mc5** levels of
[tcpl](https://github.com/USEPA/CompTox-ToxCast-tcpl), 
providing concentration-response curve fitting functionality based on [tcplfit2](https://github.com/USEPA/CompTox-ToxCast-tcplFit2).
It utilizes the [Invitrodb version 3.5 release](https://cfpub.epa.gov/si/si_public_record_Report.cfm?dirEntryId=355484&Lab=CCTE)
as its backend database.

### Use conda environment (optional)
- `conda create --name pytcpl`
- `conda activate pytcpl`
- `conda install pip`

### Install dependencies
  - `pip install -r requirements.txt`

### Navigate into python package
- `cd src`

### Run main pipeline script:
- Goto [config.yaml](config/config.yaml) for customizing pipeline behaviour
- Goto [config_db.yaml](config/config_db.yaml) for setting database login credentials
- Goto [aeid_list.in](config/aeid_list.in) for setting assay endpoints processed by pipeline

```bash 
python src/pipeline.py
```

<details><summary>(Optional) redirect terminal output to log file</summary>

```bash
python src/pipeline.py --unicode | tee export/logs/log.out
```
</details>

### Visualize fitted curves (per assay/chemical pair)
```bash
streamlit run src/app.py --server.address="localhost"
```
- The [Curve Surfer](http://localhost:8501/) web app should open as new browser tab
- Curve Surfer only works for assay endpoints already run in pipeline
- Goto official [https://streamlit.io](https://streamlit.io)

### Run ML for assay endpoint (single aeid)
- Goto [ml.ipynb](ml/ml.ipynb) for running the pipeline (jupyter notebook)
- Goto [config_ml.yaml](config/config_ml.yaml) for customizing ML pipeline behaviour


## This part is only relevant for development!
### Generate requirements.txt file with pip:
```bash
pipreqs --encoding=utf8 --force --mode no-pin"
```
- Remove duplicates and replace problem packages manually
  - `sqlalchemy`
  - `mysql-connector-python`
- Add
  - `snakeviz`

### View profiling results (if profiling was activated)
- Activate profiling by setting `apply_profiler: 1` in [config.yaml](config/config.yaml)
```bash
snakeviz export/profile/pipeline.prof
```
- The call stack with performance details should open in browser
- Goto official [SnakeViz](https://jiffyclub.github.io/snakeviz/) website

### Not working or maybe for later:
- Creates docs (there are no docstrings at all yet)
```bash
sphinx-apidoc -o ../docs .
```
```bash
cd ../docs && make clean && make html && cd ../pytcpl
```

- Build python package
```bash
python -m build
```


