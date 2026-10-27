# SoilMoisture-HSI

Reproducible analysis code for moisture-robust soil-property prediction from ASD FieldSpec hyperspectral reflectance.

## Study

The study includes 63 physical soils sampled at 21 Austrian sites and 426 repeated spectral observations acquired across a measured moisture gradient. The target of the analyses is accurate soil-property prediction across moisture conditions, including previously unseen soils. The original laboratory dataset is not distributed in this repository.

## Code currently available

- `soil_moisture_rigorous_v4.py`: stratified group-based evaluation of multi-head ResFCN variants and PLSR benchmarks.
- `soil_moisture_final_controls.py`: endpoint, one-random-moisture and all-moisture calibration strategies.
- `analyze_soil_moisture_evidence.py`: paired prediction analysis and soil-level uncertainty.

Further model scripts are being added from the complete archived research package.

## Validation principles

Repeated spectra from the same physical soil are kept in the same outer test fold for soil-independent evaluation. Sampling sites are grouped for site-independent evaluation. Transformations and tuning should be fitted within the training partition. Evaluation is based on out-of-fold predictions rather than performance on the final model's training data.

## Data and reproducibility

**Data are not included.** The experiments require the original soil reference labels and 2,151-band reflectance matrix, with sample identifiers, measured moisture content, and the original recorded soil properties. Availability of those data is subject to collaborator/institutional approval.

These scripts originated from iterative study development and are supplied for transparent computational methods. Configuration, dependency, and path checks may be needed when reproducing the work outside the original environment.

## Software

Python with packages used by the workflows, including NumPy, pandas, scikit-learn, PyTorch, matplotlib, and openpyxl. Inspect individual script imports and command-line help for exact options. No cluster-specific Slurm submission files are included.

## Citation

Please cite the associated paper once published. Until then, reference the repository and acknowledge the study authors.

## License

No open-source license is granted until the research team confirms the applicable licensing terms.
