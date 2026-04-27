Data README — TorGen
====================

The training data consists of pre-processed daily .pt.gz files
(1996–2024). Each file contains:

  - "wx":     torch.Tensor (16, 270, 270)  — daily weather input
  - "tracks": torch.Tensor (N, 6)          — tornado track vectors
  - "date":   str                          — ISO date (YYYY-MM-DD)

Track vector columns: [start_easting, start_northing, end_easting,
end_northing, width, EF_rating]. All spatial values are normalized to
[0, 1] within the model domain.

Weather tensor channels (all normalized to [0, 1]):
  0–9:   NARR reanalysis — CAPE, CIN, 0–6 km shear, storm-relative
         winds, precipitable water, 2-m RH, surface temperature,
         and additional convective parameters (daily max/min aggregates
         over the 06Z–03Z convective day)
  10–11: NLDN lightning flash density (day-of and day-prior)
  12–15: Static fields (elevation, MODIS land-use, NLCD tree canopy
         cover, UTC offset)


How to obtain the data
----------------------

The .pt.gz files are NOT included in this repository due to size
(~20 GB total). They were built by a preprocessing pipeline that
queries several proprietary and public data sources:

1. NARR Reanalysis (NOAA/NCEI)
   - Source: https://www.ncei.noaa.gov/products/weather-climate-models/north-american-regional
   - Variables: CAPE, CIN, 0–6 km shear, storm-motion winds,
     precipitable water, 2-m relative humidity, surface temperature
   - Resolution: ~32 km Lambert Conformal, regridded to 270x270
     (~10.8 km) fine grid

2. NLDN Lightning (Vaisala)
   - Source: https://www.vaisala.com/en/products/national-lightning-detection-network
   - Cloud-to-ground flash counts gridded to 0.10-degree cells
   - Daily summary data is freely available

3. NCEI Storm Events (NOAA)
   - Source: https://www.ncdc.noaa.gov/stormevents/
   - Tornado tracks: start/end coordinates, width, EF rating
   - Public dataset, queried via the Storm Events database

4. Static Fields
   - SRTM / Copernicus DEM (elevation)
   - MODIS MCD12Q1 (land-use classification)
   - NLCD Tree Canopy Cover (2021)
   - UTC offset (derived from timezone polygons)

Placing data in this folder
----------------------------

Once you have obtained the .pt.gz files, place them directly in this
data/ folder:

  data/
    1996-02-01.pt.gz
    1996-02-02.pt.gz
    ...
    2024-08-31.pt.gz

Then update the drive_dir / local_cache_dir paths in the training
notebook or config to point to this data/ directory.
