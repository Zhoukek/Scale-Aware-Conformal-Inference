#!/bin/bash
python base_test.py "configs/AMZN_test.yaml"
python base_test.py "configs/GOOGL_test.yaml"
python base_test.py "configs/daily-climate_test.yaml"
python base_test.py "configs/MSFT.yaml"

# python base_test.py "configs/elec2_test.yaml"