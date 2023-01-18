Entry Points
============

# Assumptions

- `SETTINGS.json` exists in this directory.
  * settings file name can be changed with `--settings=filename` for python script
- Test data are in `INPUT_DIR/test/`
  * INPUT_DIR is specified in SETTINGS.json
- All test data are listed in `INPUT_DIR/sample_submission.csv`
  in alphabetical order
- Templates are in `TEMPLATE_DIR/10k/fit*.h5`
- `OUTPUT_DIR` do not have same output, or use `--overwrite`.


# 1. Prepare

```bash
$ python3 src/prepare/detect_real_noise.py
```

* Creates `OUTPUT_DIR/test_real_noise.h5` which records if each test data
has flat (time- and frequency-independent) noise or real noise.


# 2. Grid search

Two types of test data, flat noise and real noise, are processed differently.

```bash
$ python3 src/power_sum/search_power_sum_optimal.py src/power_sum/wide_120_flat.yml
$ python3 src/power_sum_real/search_power_sum_real.py src/power_sum_real/wide_120_real.yml
```

* The YAML file specifies the search parameter, number of templates and range of frequency slope (time derivative)
* Outputs are saved in `OUTPUT_DIR/wide_120_flat` and `OUTPUT_DIR/wide_120_real`, respectively
* OUTPUT_DIR is specified in SETTINGS.json and the directory names are file names of the configuration file


# 3. Followup search

Follow up grid search with sinc kernel in frequency dimension and with finer grid. 

```bash
$ python3 src/confirm_power_sum/confirm_power_sum.py src/confirm_power_sum/confirm_400_flat.yml

$ python3 src/confirm_power_sum_real/confirm_power_sum_real.py src/confirm_power_sum_real/confirm_400_real.yml
```

* Outputs are saved in `OUTPUT_DIR/confirm_400_flat` and `OUTPUT_DIR/confirm_400_real`


# 4. Create submission file

Gather search results and create submission file.

```bash
$ python3 src/confirm_power_sum/submit.py confirm_400_flat
$ python3 src/confirm_power_sum_real/submit.py confirm_400_real

$ python3 src/submit/submit_significance.py src/submit/submit.yml
$ python3 src/submit/submit_probability_trivial.py src/submit/submit.yml

$ python3 src/submit/submit_final.py src/submit/submit.yml
```

- Outputs are in `OUTPUT/submit`


`run.sh` executes all of these for a very small subset of the test data.
