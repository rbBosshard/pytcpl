## This part is only relevant for development!

### Generate requirements.txt:

```bash
pipreqs --encoding=utf8 --force --mode no-pin --savepath config/requirements.txt
```

```bash
python config/handle_requirements.py
```

### View profiling results

- Activate profiling by setting `apply_profiler: 1` in [config.yaml](config/config.yaml) and
  run `python src/pipeline.py`
  The call stack with performance details should open in new browser tab:

```bash
snakeviz data/logs/pipeline.prof
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

#### Initialize empty docs

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
