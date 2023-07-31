## _pytcpl_: A lite Python version of [tcpl: An R package for processing high-throughput chemical screening data](https://github.com/USEPA/CompTox-ToxCast-tcpl)

Welcome to the GitHub repository for the _pytcpl_ package.

_pytcpl_ is a streamlined Python package that incorporates the **mc4** and **mc5** levels of
[tcpl](https://github.com/USEPA/CompTox-ToxCast-tcpl), 
providing concentration-response curve fitting functionality based on [tcplfit2](https://github.com/USEPA/CompTox-ToxCast-tcplFit2).
It utilizes the [Invitrodb version 3.5 release](https://cfpub.epa.gov/si/si_public_record_Report.cfm?dirEntryId=355484&Lab=CCTE)
as its backend database.
___


### (Optional) Use conda environment
```bash
conda create --name pytcpl
```
```bash
conda activate pytcpl
```
```bash
conda install pip
```


### Install dependencies
```bash 
pip install -r config/requirements.txt
```


### Run main pipeline script:
- Goto [config.yaml](config/config.yaml) for customizing pipeline behaviour
- Goto [config_db.yaml](config/config_db.yaml) for setting MySQL database login credentials
- Goto [aeid_list.in](config/aeid_list.in) for setting assay endpoints processed by pipeline
- Goto [DDL](config/DDL/) for setting Data Definition Language (DDL) statements, used to create new MySQL database schema objects
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
___


## This part is only relevant for development!
### Generate requirements.txt:
```bash
pipreqs --encoding=utf8 --force --mode no-pin --savepath config/requirements.txt
```
```bash
python config/handle_requirements.py
```


### View profiling results
- Activate profiling by setting `apply_profiler: 1` in [config.yaml](config/config.yaml) and run `python src/pipeline.py`
The call stack with performance details should open in new browser tab:
```bash
snakeviz export/logs/profiler/pipeline.prof
```
- Goto [SnakeViz](https://jiffyclub.github.io/snakeviz/) website


### Documentation with Sphinx
#### Update docs
```bash
docs/make clean
```
```bash
docs/make html
```

#### Initialize docs
```bash
sphinx-quickstart docs -q --project="pytcpl" --author="R. Bosshard" --release="0.1"
```
```bash
sphinx-apidoc -o docs src/
```
Add _modules_ to `index.rst`:
```
# ...

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   
 # ...
```

Edit to `conf.py`:
```
# ...

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# ...

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]

# ...

html_theme = 'sphinx_rtd_theme'
```

- Goto official [sphinx-doc.org](https://www.sphinx-doc.org) website


### Build python package
```bash
python -m build
```


### Remove conda virtual environment
```bash
conda env list
```
```bash
conda deactivate
```
```bash
conda remove --name pytcpl --all
```


