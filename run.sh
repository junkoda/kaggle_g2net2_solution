#!/bin/sh

# Script for running tiny fraction of test data
# remove --i=... for full run

# 1. Prepare
python3 src/prepare/detect_real_noise.py

# 2. Grid search
python3 src/power_sum/search_power_sum_optimal.py --i=:3 src/power_sum/wide_120_flat.yml

python3 src/power_sum_real/search_power_sum_real.py --i=:20 src/power_sum_real/wide_120_real.yml

# 3. Followup search with finer grid and sinc kernel
python3 src/confirm_power_sum/confirm_power_sum.py --i=:3 src/confirm_power_sum/confirm_400_flat.yml

python3 src/confirm_power_sum_real/confirm_power_sum_real.py --i=:20 src/confirm_power_sum_real/confirm_400_real.yml

# 4. Create submission file
python3 src/confirm_power_sum/submit.py confirm_400_flat
python3 src/confirm_power_sum_real/submit.py confirm_400_real

python3 src/submit/submit_significance.py src/submit/submit.yml
python3 src/submit/submit_probability_trivial.py src/submit/submit.yml

python3 src/submit/submit_final.py src/submit/submit.yml
