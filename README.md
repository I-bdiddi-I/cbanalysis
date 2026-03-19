# cbanalysis

**cbanalysis** is a modular, physics‑transparent Python framework for 
running multiple **Telescope Array-style pipelines** under a single unified
CLI. 

It is designed to host several independent pipelines - such as **cbspec** (UHECR
spectrum), etc. - all sharing a consistent configuration system, directory
structure, and logging model.

The framework emphasizes:
- explicit, reproducible physics logic
- YAML-driven configuration
- modular pipeline design
- publication-quality outputs

---
## Pipelines

### cbspec - UHECR Spectrum Pipeline
The `cbspec` command runs a full TA-style spectrum analysis using TASD or CBSD
surface-detector data.

It performs:
- parquet ingestion (MC + data)
- automatic tree-type detection
- TA-style quality cuts

### **Data ingestion & processing** 
- Reads **parquet** files for MC and data 
- Automatic **tree‑type detection** (`resTree` vs `tTlfit`) 
- TA‑style **quality cuts** (fully configurable in YAML)
- Batch‑wise parquet processing with detailed logging 
- FD energy correction and log10(E/eV) computation 
- Extracts: 
  - MC reconstructed log10(E) 
  - MC thrown log10(E) 
  - Data reconstructed log10(E) 

---

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

---

## **Outputs** 

### **Plots**
Saved to **two locations**:
- **Global copies** → `output/plots/` 
- **Run-specific copies** → `output/runs/<timestamp>/plots/` 

#### **Plots saved**:
- `{array_type}_aperture.png` 
- `{array_type}_exposure.png` 
- `{array_type}_flux.png` 
- `{array_type}_spectrum.png` 
- MC/data histograms:
  - `{array_type}_mc_reco_hist.png`
  - `{array_type}_mc_thrown_hist.png`
  - `{array_type}_data_hist.png`

(Names are array-tagged: TASD or CBSD)

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