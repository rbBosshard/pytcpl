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
pip install -r requirements.txt
```


### Run main pipeline script:
- Goto [config.yaml](config/config.yaml) for customizing pipeline behaviour
- Goto [config_db.yaml](config/config_db.yaml) for setting MySQL database login credentials
- Goto [aeid_list.in](config/aeid_list.in) for setting assay endpoints to be processed by pipeline
- Goto [DDL](config/DDL) for setting Data Definition Language (DDL) statements, used to create new MySQL database schema objects
```bash 
python src/pipeline.py
```

<details><summary>(Optional) redirect terminal output to log file</summary>

```bash
python src/pipeline.py --unicode | tee export/logs/log.out
```
</details>

- Goto [logs](logs) to see the redirected terminal logs and check in `error.out` what went wrong for which assay endpoint id

### Visualize fitted curves (on assay endpoint & chemical pair basis)
```bash
streamlit run src/app.py --server.address="localhost"
```
- The [Curve Surfer](http://localhost:8501/) web app should open as a new browser tab
- Curve Surfer only works for assay endpoints already processed by pipeline
- Goto official [https://streamlit.io](https://streamlit.io) website
