### pytcpl: A Simplified Python version of  [tcpl: An R package for processing high-throughput chemical screening data](https://github.com/USEPA/CompTox-ToxCast-tcpl)

Welcome to the GitHub repository for the pytcpl package.

The pytcpl package provides a set of tools for processing and modeling high-throughput and high-content chemical screening data of the US EPA ToxCast program. 

### Activate conda environment
`conda activate pytcpl`

### Main run script:
`python pytcpl/pipeline.py`

### Config file for database login
`pytcpl/config.yaml`

### If not already, navigate to the directory where you can find all source files
`cd pytcpl`

### Create docs
- `sphinx-apidoc -o ../docs .`
- `cd ../docs && make html && cd ../pytcpl`

### Build python package
`python -m build`

<details>
  <summary>requirements</summary>

 ### Generate a requirements.txt file with pip:
`pip freeze > requirements.txt`

### Install packages with pip from requirements.txt:
`pip install -r requirements.txt`

</details>
