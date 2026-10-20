#!/usr/bin/env bash
set -euo pipefail

npm install
python -m pip install --upgrade pip
python -m pip install -r tools/spark/requirements.txt
