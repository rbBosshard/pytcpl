## _pytcpl_: A lite Python version of [tcpl: An R package for processing high-throughput chemical screening data](https://github.com/USEPA/CompTox-ToxCast-tcpl)

Welcome to the GitHub repository for the _pytcpl_ package.

_pytcpl_ is a streamlined Python package that incorporates the **mc4** and **mc5** levels of
[tcpl](https://github.com/USEPA/CompTox-ToxCast-tcpl), 
providing concentration-response curve fitting and hitcalling based on [tcplfit2](https://github.com/USEPA/CompTox-ToxCast-tcplFit2).
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

### Before running main pipeline script:
#### Required
- Goto [config_db.yaml](config/config_db.yaml) to add/set MySQL your database login credentials
- Goto [aeid_list.in](config/aeid_list.in) to set assay endpoint ids (aeids) to be processed by the pipeline
#### Optional
- Goto [config.yaml](config/config.yaml) to customize pipeline behaviour
- Goto [DDL](config/DDL) to set Data Definition Language (DDL) statements, used to create new MySQL database schema objects
#### Run pipeline
```bash 
python src/pipeline.py
```

<details><summary>(Optional) redirect terminal output to log file</summary>

```bash
python src/pipeline.py --unicode | tee export/logs/log.out
```

- Goto [logs](logs) to see the redirected terminal logs and check in `error.out` what went wrong for which assay endpoint id
</details>

### _Curve surfer_: Inspect fitted curves, hitcall labels and potency estimates
```bash
streamlit run src/app/curve_surfer.py --server.address="localhost"
```
- If you run the command, the [Curve Surfer](http://localhost:8501/) web app should open as a new browser tab
- Curve Surfer only works for assay endpoints already processed by pipeline
- Goto official [https://streamlit.io](https://streamlit.io) website

[streamlit-curve_surfer-2023-08-09-22-08-26.webm](https://github.com/rbBosshard/pytcpl/assets/100019212/0578d442-826b-4c78-b95e-9f0447408123)
