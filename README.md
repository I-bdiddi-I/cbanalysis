# cbanalysis

**cbanalysis** is a modular, physics‑transparent Python framework for 
running multiple **Telescope Array-style pipelines** under a single unified
CLI. 

It is designed to host several independent pipelines - such as **cbprocess** 
(energy reconstruction + qaulity cuts), **cbspec** (UHECR
spectrum), and **cbefficiency**
(trigger/reconstruction efficiency)- all sharing: 
- a consistent YAML-driven configuration system 
- a unified output directory structure
- a common logging model
- explicit, reproducible physics logic

The framework emphasizes:
- modular pipeline design
- physics transparency
- publication-quality outputs
- clean separation o orchestration, physics, plotting, and I/O

---
## Pipline Overview & Data Flow
A core design principle of **cbanalysis** is that pipelines for a directo workflow:
```
cbprocess → cbspec
           → cbefficiency
```

### **cbprocess is the authoritative preprocessing pipeline** 
It is responsible for:
- reading MC + data parquet files
- applying TA-style quality cuts
- reconstructing log10(E/eV)
- splitting into time periods
- producing all standard CSV outputs

These CSVs for the canonical inputs for downstream pipelines.

### **cbspec and cbefficiency consume cbprocess outputs** 
By default:
- **cbspec** reads the reconstructed and thrown-energy CSVs produced by cbprocess
- **cbefficiency** reads the thrown-energy CSVs produced by cbprocess

Users *may* override these paths in YAML or via CLI, but the recommended workflow is:
```
cbprocess → cbspec
cbprocess → cbefficiency
```
This ensures consistent cuts, reconstruction and period definitions across all physics
products.
---
## Pipelines

---
### cbprocess - Energy Reconstruction & Quality Cuts Pipeline
The `cbprocess` command performs the full TA-style preprocessing workflow and produces
**canonical CSV files** used by all downstream pipelines.

### **Data ingestion & processing** 
- Reads **parquet** files for MC and data
- Automatic **tree‑type detection** (`resTree` vs `tTlfit`)
- Applies TA‑style **quality cuts** (fully configurable in YAML)
- Batch‑wise parquet processing with detailed logging 
- Reconstructs log10(E/eV) with FD energy corrections
- Extracts: 
  - MC reconstructed log10(E/eV)  (full cuts)
  - MC thrown log10(E/eV) (no cuts / geom cuts / full cuts)
  - Data reconstructed log10(E/eV) (full cuts)

### **Period splitting**
- Optional splitting into **N time periods**
- Each period receives its own set of CSV outputs
- Period ranges logged and saved

### **Outputs**
cbprocess produces the **authoritative energy CSVs** consumed by cbspec 
and cbefficiency:
```
{array}_mc_recon_cut.csv
{array}_data_recon_cut.csv
{array}_mc_thrown_nocuts.csv
{array}_mc_thrown_geomcuts.csv
{array}_mc_thrown_fullcuts.csv
```
Saved to:
- `output/cbprocess/data/`
- `output/cbprocess/runs/<timestamp>/data/`

This pipeline **does not produce plots** (under normal functionality), 
so only `data/` directories are created.

---
### cbspec - UHECR Spectrum Pipeline
The `cbspec` command runs a full TA-style spectrum analysis using TASD or CBSD
surface-detector data.

### **Data ingestion & processing** 
- Reads **CSV files produced by cbprocess**
- Automatically aligns energy arrays and period ranges

All physics-agnostic preprocessing is handled upstream by cbprocess


### **Physics pipeline** 

#### **Energy binning**
- Energy binning in **log10(E/eV)**
- Histograms MC_recon, MC_thrown, and data
- Bin filtering:
  - log10(E/eV) > 18.5
  - N_MC_thrown > 1
- Convert energies from **log10(E/eV)** to **eV** for:
  - $E_i$
  - $\Delta E_i$

#### **Aperture** 

$$
\alpha_i = \frac{(N^{\text{MC}}_{\text{REC}})_i}{(N^{\text{MC}}_{\text{GEN}})_i} \, A_{\text{GEN}} \, \Omega_{\text{GEN}} 
$$ 

#### **Exposure** 

$$ 
\lambda_i = \alpha_i \times T 
$$ 

#### **Flux**

$$
J_i = \frac{(N^{\text{DATA}}_{\text{REC}})_i / \Delta E_i}{\lambda_i} 
$$

#### **Feldman–Cousins confidence intervals** 
- Uses **FCpy** (NIST)
- Compute lower/upper bounds on counts
- Propagated to flux and spectrum

#### **Spectrum** 
$$
S_i = E_i^3 J_i 
$$ 

## **Outputs** 

Saved to:
- `output/cbspec/plots/` 
- `output/cbspec/data/` 
- `output/cbspec/runs/<timestamp>/plots/`
- `output/cbspec/runs/<timestamp>/data/`

Includes:
- aperture, exposure, flux, spectrum plots
- MC/data histograms
- flux CSV
- spectrum CSV

---
### cbefficiency - Trigger & Reconstruction Efficiency Pipeline
The `cbefficiency` command computes TA-style trigger/reconstruction efficiency 
curves using **MC thrown and reconstructed energies produced by cbprocess**.

### **Data ingestion & processing** 
- Reads **cbprocess-generated CSVs** for:
  - MC thrown (geomcuts)
  - MC thrown (full cuts)
- Optional **period splitting**
- Computes log10(E/eV) distributions
- Aligns thrown and reconstructed energy bins

### **Physics pipeline** 

#### **Efficiency computation**
$$
\epsilon(E) = \frac{(N^{\text{MC}}_{\text{REC}})}{(N^{\text{MC}}_{\text{GEN}})}
$$ 
- Efficiency vs. log10(E/eV)
- Binomial uncertainties
- Optional smooting

#### **Logistic fitting** 


#### **Flux**

$$
J_i = \frac{(N^{\text{DATA}}_{\text{REC}})_i / \Delta E_i}{\lambda_i} 
$$

#### **Feldman–Cousins confidence intervals** 
- Uses **FCpy** (NIST)
- Compute lower/upper bounds on counts
- Propagated to flux and spectrum

#### **Spectrum** 
$$
S_i = E_i^3 J_i 
$$ 

## **Outputs** 

Saved to:
- `output/cbspec/plots/` 
- `output/cbspec/data/` 
- `output/cbspec/runs/<timestamp>/plots/`
- `output/cbspec/runs/<timestamp>/data/`

Includes:
- aperture, exposure, flux, spectrum plots
- MC/data histograms
- flux CSV
- spectrum CSV

---

### **Data products** 
Saved to **two locations**:
- **Global copies** → `output/data/`
- **Run-specific copies** → `output/runs/<timestamp>/data/`

#### **Flux CSV**
`{array_type}_flux.csv` 
``` 
Energy, Bin_size, N_events, Exposure, J, Lower, Upper 
``` 

#### **Spectrum CSV**
`{array_type}_spectrum.csv`
``` 
Energy, Spectrum, Lower, Upper 
``` 

---

## Configuration
All configuration is YAML‑driven 

`config/default_config.yaml`

Contains:
- array type + file paths
- energy bin edges
- generated area + solid angle
- runtime
- quality cuts
- output directory structure

---

## CLI Usage

### Default run
```bash
python -m cbspec
```
### CLI overrides
```bash
python -m cbspec \ 
  --config config/default_config.yaml \
  --array TASD \
  --mc-file /path/to/mc.parquet \
  --dt-file /path/to/data.parquet
```

---
## Installation

### 1. Create the conda environment 

```bash 
conda env create -f environment.yml 
conda activate cbspec
```

### 2. Install cbspec in editable mode
```bash
pip install -e .
```

---
 
## Project Structure
```
src/cbspec/
    __init__.py
    __main__.py
    binning.py
    cli.py 
    constants.py
    data_classes.py
    exposure.py 
    feldman_cousins.py
    flux.py
    load_config.py
    logging_utils.py
    main.py 
    output_utils.py
    plotting.py 
    process_data.py
    spectrum.py 
```

---

## Pipeline Summary

1. Load configuration
2. Create timestamp run directory
3. Read MC + data parquet files 
4. Apply TA‑style quality cuts 
5. Build log10(E/eV) energy bins 
6. Histogram MC_reco, MC_thrown, and data 
7. Compute aperture $\alpha$(E)
8. Compute exposure $\lambda$(E)
9. Compute Feldman–Cousins intervals 
10. Compute flux J(E)
11. Compute spectrum E$^3$J(E)
12. Save CSVs (global + run-specific)
13. Produce publication‑quality plots (global + run-specific)
11. Log all steps to text + JSONL (global + run-specific)

---

## Developer Notes
- Modular, physics-transparent code
- Explicit variable naming
- No hidden state
- No silent cuts
- Easy to extend with new physics models

Adding new physics:
- `process_data.py` for new tree types
- `binning.py` for new binning schemes
- `exposure.py` for new geometry models
- `flux.py` for alternative flux definitions

Adding new outputs:
- use `output_utils.py` to keep output logic centralized

---
## License
MIT License

© 2026 Robert D’Avignon