# TorGen: Synthetic Tornado Outbreak Generation with MDN-CVAE

A conditional variational autoencoder with a mixture density network decoder that generates variable-length sets of tornado track vectors conditioned on daily weather. Different latent samples from the same weather produce different plausible outbreak realizations, forming the basis for a synthetic tornado catalog.

## Goals

1. **Stochastic outbreak modeling** -- Generate realistic tornado track sets that capture the full distribution of possible outcomes for a given weather environment.
2. **Synthetic catalog generation** -- Sample thousands of plausible tornado seasons for catastrophe modeling and risk assessment.
3. **Physical grounding** -- Cross-attention to spatially resolved weather features ensures geographic plausibility.

## Data

- **Input:** 16-channel daily weather tensor (270x270 grid, ~10.8 km spacing) covering the central/eastern CONUS tornado corridor. Channels include NARR convective parameters, NLDN lightning density, and static fields (elevation, land use, tree cover). All normalized to [0, 1].
- **Target:** Variable-length sets of tornado track vectors (start position, bearing, length, width, EF rating) from NCEI Storm Events. 0 to ~330 tracks per day.
- **Scope:** 1996--2024. Train on 1996--2018, validate on 2019--2021, test on 2022--2024.

See `data/readme_data.txt` for details on data sources and how to obtain the pre-processed `.pt.gz` files.

## Requirements

Python >= 3.10. Install dependencies:

```bash
pip install torch>=2.0 scipy pandas pyarrow matplotlib
```

Or install the package directly:

```bash
pip install git+https://github.com/jcaramichaellehigh/TorGen.git
```

## How to Run

Training and evaluation are done via the **`notebooks/train_mdn.ipynb`** notebook, designed to run on Google Colab with a GPU runtime.

### Quick start

1. Open `notebooks/train_mdn.ipynb` in Google Colab.
2. The first cell installs the package from GitHub.
3. Mount Google Drive when prompted -- the notebook expects data files and writes checkpoints to Drive.
4. Update the `drive_dir` path in the `MDNTrainConfig` to point to your `.pt.gz` data directory (see `data/readme_data.txt`).
5. Run all cells. Training, loss curves, sample visualizations, and count/EF distribution diagnostics are produced inline.

### Running locally

```bash
git clone https://github.com/jcaramichaellehigh/TorGen.git
cd TorGen
pip install -e ".[all]"
jupyter notebook notebooks/train_mdn.ipynb
```

Update `drive_dir` in the config cell to point to your local `data/` directory containing the `.pt.gz` files.

## Repository Structure

```
TorGen/
├── data/
│   └── readme_data.txt          # Data sources and setup instructions
├── notebooks/
│   └── train_mdn.ipynb          # Training and evaluation notebook
├── src/torgen/
│   ├── data/dataset.py          # Dataset and collation
│   ├── models/                  # MDN-CVAE architecture
│   ├── training/                # Training loop and config
│   ├── sampling.py              # Outbreak sampling from MDN params
│   └── viz/                     # Visualization utilities
├── pyproject.toml
└── README.md
```

## License

Academic use only.