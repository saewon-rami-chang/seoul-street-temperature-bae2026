# Column reference — `seoul_sdot_features_<year>.parquet`

207 columns: 9 identifiers / targets + 198 model features (the feature set of the paper, grouped as in Table 2). One row = one S-DoT sensor × one hour (KST). Spatial features are sampled at the sensor coordinate by bilinear interpolation of 10 / 100 / 250 / 1,000 m rasters (EPSG:5179), so every spatial band appears four times with the suffix `_10m`, `_100m`, `_250m` or `_1000m`. Missing values are stored as null; the models use XGBoost's native missing-value handling. Model features are float32.

## Identifiers, coordinates and targets

| Column | Type | Unit | Resolution | Meaning |
|---|---|---|---|---|
| `sensor_id` | large_string | – | – | S-DoT serial number |
| `timestamp` | timestamp[us] | KST | hourly | Hour of observation, Korea Standard Time (UTC+9), stored as a naive timestamp |
| `hardware_generation` | large_string | – | per sensor | `V02Q` or `OC3`, from the serial-number prefix |
| `loro_region` | large_string | – | per sensor | Leave-one-region-out holdout region `A`–`F` (sensor inside the region box) or `none`; the 1 km buffers are computed in code |
| `lon` | double | degree (WGS84) | per sensor | Sensor longitude |
| `lat` | double | degree (WGS84) | per sensor | Sensor latitude |
| `aws_baseline` | double | °C | hourly, sensor location | IDW of the 15 nearest KMA AWS stations (p = 2), lapse-rate corrected (Γ = −6.5 °C km⁻¹); T_base in Eq. 2–3 |
| `sdot_temp` | double | °C | hourly | Observed S-DoT air temperature after QC (= aws_baseline + target_resid) |
| `target_resid` | double | °C | hourly | Model target: observed − baseline (Eq. 1). Training uses rows with |target_resid| < 25 °C |

## Morphology — 33 bands × 4 resolutions = 132 features (Table A.2)

Source: NGII 1:1,000 Digital Topographic Map v2.0 (VWorld), rasterised at 1 m and block-aggregated to each resolution.

| Band (`<band>_<res>`) | Unit | Meaning |
|---|---|---|
| `DEM` | m a.s.l. | Terrain elevation (TIN from NGII contours and spot heights) |
| `Slope` | degree | Terrain gradient from the DEM |
| `Aspect_Cos` | – | Cosine of terrain aspect (−1 north-facing, +1 south-facing) |
| `Aspect_Sin` | – | Sine of terrain aspect (−1 west-facing, +1 east-facing) |
| `SVF` | – | Sky view factor, 36-ray cast, 200 m range on the 1 m DSM (0 enclosed, 1 open) |
| `Canopy_Height` | m | Maximum obstacle height above ground (buildings, bridges, walls, fences, levees, overpasses; trees excluded) |
| `Building_Fraction` | fraction | Share of the cell covered by building footprints |
| `Building_Height` | m | Mean building height (floors × type-specific storey height + roof offset); missing where no buildings |
| `Building_Height_SD` | m | Standard deviation of building heights in the cell; missing where no buildings |
| `Apartment_Fraction` | fraction | Share covered by high-rise residential buildings (≥5 floors) |
| `MidRes_Fraction` | fraction | Share covered by mid-rise residential buildings (villa / multi-family) |
| `LoRes_Fraction` | fraction | Share covered by low-rise detached housing (<4 floors) |
| `NonRes_Fraction` | fraction | Share covered by commercial, public and industrial buildings |
| `Lightweight_Fraction` | fraction | Share covered by lightweight structures (temporary buildings, greenhouses) |
| `Building_Fraction_NS` | fraction | Share covered by buildings with long axis oriented N–S (45°–135°) |
| `Building_Fraction_EW` | fraction | Share covered by buildings with long axis oriented E–W |
| `FAI_NS` | – | Frontal area index for north–south wind |
| `FAI_EW` | – | Frontal area index for east–west wind |
| `Road_Fraction` | fraction | Share covered by road surfaces and intersections |
| `Sidewalk_Fraction` | fraction | Share covered by sidewalks, crosswalks, safety islands, stairs |
| `Water_Fraction` | fraction | Share covered by rivers, lakes, reservoirs, wetlands |
| `Impervious_Fraction` | fraction | Share covered by all impervious surfaces |
| `Vegetation_Fraction` | fraction | Share covered by vegetation polygons and tree canopy (3 m buffer) |
| `Road_Lane_Density` | lane-m m⁻² | Total lane length per unit area from road centre lines with lane counts |
| `Dist_To_Water` | m | Euclidean distance to the nearest water body ≥1 ha |
| `Dist_To_Vegetation` | m | Euclidean distance to the nearest vegetation patch ≥0.25 ha |
| `Bare_Soil_Fraction` | fraction | Share covered by cut-and-fill earthwork |
| `Canyon_Width_NS` | m | North–south street-canyon width (both opposing rays hit an obstacle within 60 m) |
| `Canyon_Width_EW` | m | East–west street-canyon width |
| `Canyon_Fraction_NS` | fraction | Share of open pixels forming valid N–S canyons |
| `Canyon_Fraction_EW` | fraction | Share of open pixels forming valid E–W canyons |
| `Street_Tree_Density` | trees ha⁻¹ | Point count of individual street trees per hectare |
| `Subway_Vent_Fraction` | fraction | Share covered by subway ventilation structures |

## Land cover — 6 super-classes × 4 resolutions = 24 features

Source: Ministry of Environment Level-3 land cover map (EGIS), ~1 m, aggregated to six super-classes. Unit: fraction of the cell.

| Feature (`LC_<class>_<res>`) | Meaning |
|---|---|
| `LC_Urban` | Share of urban / built-up classes |
| `LC_Agri` | Share of agricultural classes |
| `LC_Forest` | Share of forest classes |
| `LC_Grass` | Share of grassland classes |
| `LC_Water` | Share of water classes |
| `LC_Barren` | Share of barren classes |

## Shadow fraction — 4 features

| Feature | Unit | Resolution | Meaning |
|---|---|---|---|
| `shadow_fraction_<res>` | fraction | 10 / 100 / 250 / 1,000 m, hourly | Share of ground shaded by buildings at the current sun position (look-up table of 1,186 sun positions computed on the 1 m DSM) |

## Rolling / lag — 25 features (Table A.4)

Hourly, from the baseline temperature and IDW wind at the sensor location, or from ASOS 108 / the nearest AWS station as noted.

| Feature | Unit | Meaning |
|---|---|---|
| `aws_temp_lag_1h` | °C | Baseline temperature 1 h earlier |
| `aws_temp_lag_2h` | °C | Baseline temperature 2 h earlier |
| `aws_temp_lag_3h` | °C | Baseline temperature 3 h earlier |
| `aws_temp_lag_6h` | °C | Baseline temperature 6 h earlier |
| `aws_temp_mean_6h` | °C | 6 h rolling mean of the baseline temperature |
| `aws_temp_mean_24h` | °C | 24 h rolling mean of the baseline temperature |
| `aws_temp_trend_1h` | °C | Baseline temperature change over 1 h (t − t−1) |
| `aws_temp_trend_3h` | °C | Baseline temperature change over 3 h |
| `aws_temp_trend_24h` | °C | Baseline temperature change over 24 h |
| `daily_temp_swing` | °C | 24 h max − min of the baseline temperature |
| `wind_u_mean_3h` | m s⁻¹ | 3 h mean eastward wind (IDW) |
| `wind_u_mean_6h` | m s⁻¹ | 6 h mean eastward wind (IDW) |
| `wind_v_mean_3h` | m s⁻¹ | 3 h mean northward wind (IDW) |
| `wind_v_mean_6h` | m s⁻¹ | 6 h mean northward wind (IDW) |
| `solar_sum_6h` | MJ m⁻² | 6 h cumulative solar radiation (ASOS 108) |
| `solar_sum_12h` | MJ m⁻² | 12 h cumulative solar radiation (ASOS 108) |
| `cloud_mean_3h` | 1/10 | 3 h mean low/mid cloud cover (ASOS 108) |
| `cloud_mean_24h` | 1/10 | 24 h mean low/mid cloud cover (ASOS 108) |
| `precip_sum_3h` | mm | 3 h cumulative precipitation (nearest AWS) |
| `precip_sum_24h` | mm | 24 h cumulative precipitation (nearest AWS) |
| `humidity_mean_3h` | % | 3 h mean relative humidity (nearest AWS) |
| `humidity_mean_12h` | % | 12 h mean relative humidity (nearest AWS) |
| `dew_point_mean_3h` | °C | 3 h mean dew point (ASOS 108) |
| `dew_point_mean_12h` | °C | 12 h mean dew point (ASOS 108) |
| `pressure_trend_3h` | hPa | 3 h change in sea-level pressure (ASOS 108) |

## Meteorological — 9 features

| Feature | Unit | Resolution | Meaning |
|---|---|---|---|
| `wind_u_interp` | m s⁻¹ | hourly, IDW of AWS (k = 15) | Eastward wind component |
| `wind_v_interp` | m s⁻¹ | hourly, IDW of AWS (k = 15) | Northward wind component |
| `wind_speed_reg` | m s⁻¹ | hourly, from the IDW u/v | Wind speed √(u² + v²) |
| `humidity` | % | hourly, nearest AWS station | Relative humidity |
| `precipitation` | mm h⁻¹ | hourly, nearest AWS station | Precipitation |
| `solar_radiation` | MJ m⁻² | hourly, ASOS 108 (city-wide) | Global solar radiation; empty at night by KMA convention |
| `low_mid_cloud_cover` | 1/10 | hourly, ASOS 108 (city-wide) | Low and mid-level cloud cover |
| `dew_point` | °C | hourly, ASOS 108 (city-wide) | Dew-point temperature |
| `sea_level_pressure` | hPa | hourly, ASOS 108 (city-wide) | Sea-level pressure |

## Temporal — 4 features

| Feature | Unit | Meaning |
|---|---|---|
| `diurnal_phase` | – | cos(2π (h − 14) / 24); peaks at 14:00 KST |
| `calendar_phase` | – | cos(2π (d − 30) / 365.25), d = day of year |
| `solar_elev` | degree | Solar elevation (pvlib) at the sensor and hour |
| `ghi_clearsky` | W m⁻² | Clear-sky global horizontal irradiance (pvlib) |
