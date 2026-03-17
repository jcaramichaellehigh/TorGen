# TorGen: Synthetic Tornado Outbreak Generation with DETR-CVAE

A conditional variational autoencoder with a DETR-style set prediction decoder that generates variable-length sets of tornado track vectors conditioned on daily weather. Different latent samples from the same weather produce different plausible outbreak realizations, forming the basis for a synthetic tornado catalog.

## Goals

1. **Stochastic outbreak modeling** -- Generate realistic tornado track sets that capture the full distribution of possible outcomes for a given weather environment.
2. **Synthetic catalog generation** -- Sample thousands of plausible tornado seasons for catastrophe modeling and risk assessment.
3. **Physical grounding** -- Cross-attention to spatially resolved weather features ensures geographic plausibility.

## Data

- **Input:** 16-channel daily weather tensor (270x270 grid, ~10.8 km spacing) covering the central/eastern CONUS tornado corridor. Channels include NARR convective parameters, NLDN lightning density, and static fields (elevation, land use, tree cover). All normalized to [0, 1].
- **Target:** Variable-length sets of tornado track vectors (start position, bearing, length, width, EF rating) from NCEI Storm Events. 0 to ~330 tracks per day.
- **Scope:** March--June, 1996--2024 (~3,480 days). Train on 1996--2018, validate on 2019--2021, test on 2022--2024.

## License

Academic use only.