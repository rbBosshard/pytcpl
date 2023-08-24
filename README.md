## _pytcpl_: A lite Python version of [tcpl: An R package for processing high-throughput chemical screening data](https://github.com/USEPA/CompTox-ToxCast-tcpl)

#### Welcome to the GitHub repository for the _pytcpl_ package.

#### _pytcpl_ is a streamlined Python package that incorporates the **mc4** and **mc5** levels of [tcpl](https://github.com/USEPA/CompTox-ToxCast-tcpl), providing concentration-response curve fitting and hitcalling based on [tcplfit2](https://github.com/USEPA/CompTox-ToxCast-tcplFit2). It utilizes the [Invitrodb version 3.5 release](https://cfpub.epa.gov/si/si_public_record_Report.cfm?dirEntryId=355484&Lab=CCTE) as its backend database.
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

### Running pipeline:

#### Required

- Goto [config_db.yaml](config/config_db.yaml) to add/set your MySQL database connection parameters
- Goto [aeid_list.in](config/aeid_list.in) to set assay endpoint ids (aeids) to be processed by the pipeline
  
#### Optional

- Goto [config.yaml](config/config.yaml) to customize pipeline behaviour
- Goto [DDL](config/DDL) to set Data Definition Language (DDL) statements, used to create new MySQL database schema
  objects
- [Automate](src/pipeline/pipeline_setup.py) list generation [aeid_list.in](config/aeid_list.in) suited for balanced workloads for distributed compute instances 

#### Run pipeline

```bash 
python pytcpl/src/pipeline.py --instance_id 0 --instances_total 1
```

- Goto [logs](logs) to see the redirected terminal logs and check in the error logs what went wrong for which assay
  endpoint id
- Note: Parameters `instance_id` and `instances_total` are used for distributing workload onto multiple compute engine
  instances (e.g. gcloud VMs).
- Example: Distribute [workload](config/aeid_list.in) onto 2 VM instances. Run:
    - `pytcpl/src/pipeline.py --instance_id 0 --instances_total 2` on one machine and
    - `pytcpl/src/pipeline.py --instance_id 1 --instances_total 2` on another machine
  
#### Further automation steps
- [Generate](src/pipeline/pipeline_wrapup.py) per-chemical-results from all processed assay endpoints
- [Fetch](src/utils/generating_code/fetch_db_tables.py) relevant metadata tables from invitrodb
- [Remove](src/utils/generating_code/remove_output_files_not_in_aeid_list.py) output files NOT contained in [aeid_list.in](config/aeid_list.in)

### _Curve surfer_: Inspect fitted curves, hitcall labels and potency estimates

```bash
streamlit run src/app/curve_surfer_app.py --server.address="localhost"
```

- If you run the command, the [Curve Surfer](http://localhost:8501/) web app should open as a new browser tab
- Curve Surfer only works for assay endpoints already processed by pipeline
- Goto [https://pytcpl.streamlit.app/](https://pytcpl.streamlit.app/)

[streamlit-curve_surfer-2023-08-09-22-08-26.webm](https://github.com/rbBosshard/pytcpl/assets/100019212/0578d442-826b-4c78-b95e-9f0447408123)
