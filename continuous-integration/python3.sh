#!/bin/bash

# This script runs a passing subset of tests under Python 3.

python setup.py install

python -c "import sunpy.data"
python -c "import sunpy.data; sunpy.data.download_sample_data()"
python -c "import sunpy.data.sample"

python setup.py test -P util
python setup.py test -P time
python setup.py test -P map
python setup.py test -P io
python setup.py test -P image
python setup.py test -P sun

