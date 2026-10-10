# Storage and capacity rather than volume govern the water needs of United States data centers

Analysis code, derived data and figures for the study of the water that projected United States data centers need,
treated as an infrastructure capacity problem: hourly cooling-model output at every projected site, storage-yield
analysis of withdrawal demand, capacity-aware siting against existing community water systems, water-power peak
coincidence at the hours of the grid peak, and the least-cost comparison of capacity expansion, on-site storage and
dry backup.

Authors: Zainab Ashkanani (corresponding, ashkanani@tamu.edu), Konstantinos Pappas, Rabi Mohtar; Texas A&M University.

## Layout

| Folder | Content |
|---|---|
| `code/` | The analysis pipeline, in run order (see below). |
| `data/inputs/` | Small processed inputs: the 16,645 site records of the 20 IM3 runs, the climate-cell lists, the county to balancing-authority map, the ISD station list, and the per-cell summaries of the cooling-model runs (POWER and ERA5-Land). |
| `data/results/` | Every derived table of the final run: storage-yield curves and simulation results by cell, siting, utility exposure, peak coincidence, least-cost grid, sensitivities and Monte Carlo draws, validation, and `numbers_v5.json`, the single file from which every number in the manuscript is read. |
| `data/supplementary_data/` | The same results packaged as Supplementary Data 1 to 7 of the manuscript. |
| `figures/` | Figures 1 to 6 and Supplementary Figures 1 to 7, each as PNG (300 dpi) and EPS. |

## Pipeline (`code/`)

1. `fetch_power.py`, `fetch_points.py`, `fetch_climate.py`: hourly weather 2014 to 2023 from NASA POWER (every cell
   that hosts a projected site; the validation stations and measured facilities) and from Open-Meteo ERA5-Land.
2. `compute_hourly.py`, `compute_era.py`, `model_points.py`: run the cooling model (`model_ext.py`, which extends the
   published thermodynamic model with a dry condenser mode) hour by hour for warming offsets of 0, +1 and +2 C.
3. `update_dry_v5.py`: dry condenser mode at a conventional 6 K range with the COP read at the leaving condenser-water
   temperature, the bounding conventions, and the dry-cooler sizing.
4. `infra_v5.py`: storage-yield analysis, daily simulation with dry backup, capacity-aware siting, utility exposure,
   peak coincidence against EIA-930 demand, least-cost analysis with sensitivities and Monte Carlo.
5. `validate_v5.py`, `analysis.py`, `warming_rh.py`: station and facility validation, national totals, constant-RH warming.
6. `tables_v5.py`: writes `numbers_v5.json`; `figs_v5.py`, `figs_si_v5.py`: the figures.

The cooling model itself (`revmodel.py`) is the separate repository
[MTX_Climate_Datacenters](https://github.com/SAAAHco/MTX_Climate_Datacenters); `model_ext.py` imports it from a local
path set at the top of the file. Paths in the scripts point at the authors' working folders and must be edited to run.

## Raw inputs not included (public, fetched by the scripts or downloaded by hand)

- IM3 Projected US Data Center Locations v1.1, https://doi.org/10.57931/3020186
- NASA POWER hourly data, https://power.larc.nasa.gov; Open-Meteo ERA5-Land, https://open-meteo.com
- NOAA ISD-Lite station records, https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/
- EPA community water system service-area boundaries v3.0, https://github.com/USEPA/ORD_SAB_Model
- USGS public-supply water use reanalysis, https://doi.org/10.5066/P9FUL880; USGS 2015 county data, https://doi.org/10.5066/F7TB15V5
- Census cartographic boundary files 2024; EIA-930 hourly balance files 2022 and 2023; EIA Form 861, 2023

## Environment

Python 3.13 with numpy, pandas, shapely, pyogrio, matplotlib (see `requirements.txt`).

## License and citation

Code under the MIT License; data and figures under CC BY 4.0. Please cite the article and this repository
(`CITATION.cff`).
