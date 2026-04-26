# polygon

Joins ~130M Microsoft building centroids against ~33k US ZIP code polygons and
prints the top 5 ZIPs by building count. The interesting piece is
`MultiPolygonIndex` in `src/main.rs` — an `rstar` R-tree of bounding boxes
layered over one `IntervalTreeMultiPolygon` per polygon. Builds once, queries
in parallel.

## Run

```
cargo run --release
```

Expects two files under `data/`:

- `data/us-zip-codes.parquet` — GeoParquet, one row per ZIP, columns `zipcode`
  (string) + a GeoArrow `MultiPolygon` geometry column.
- `data/microsoft-buildings_point.parquet` — plain Parquet (not GeoParquet),
  one row per building, column `geometry` of WKB `Point` bytes.

Neither is a canonical published file; you build them once from the sources
below. `scripts/fetch_data.sh` automates the ZCTA step and runs the DuckDB
recipe for the buildings step (you provide the source via `$BUILDINGS_SRC`).

## Building `data/us-zip-codes.parquet`

Source: US Census Bureau **ZCTA5** TIGER/Line shapefile. Download the latest
ZCTA5 nation-wide shapefile from the Census and convert with GDAL ≥ 3.5:

```
ogr2ogr -f Parquet data/us-zip-codes.parquet \
  -sql "SELECT ZCTA5CE20 AS zipcode FROM tl_2020_us_zcta520" \
  tl_2020_us_zcta520.shp
```

The `SELECT ... AS zipcode` rename matters — `load_zipcodes` looks the column
up by that exact name. If your shapefile is the 2010 vintage, use
`ZCTA5CE10` instead.

Census ZCTA page: https://www.census.gov/programs-surveys/geography/guidance/geo-areas/zctas.html

## Building `data/microsoft-buildings_point.parquet`

Source: Microsoft's open US building footprints. The HDX repackaging on
Source Cooperative is the easiest cloud-native form:
https://source.coop/hdx/microsoft-open-buildings — filter to USA partitions.
The original per-state GeoJSON release also works:
https://github.com/microsoft/USBuildingFootprints.

The driver wants centroids as WKB Points in a plain Parquet column called
`geometry`. DuckDB streams the USA partition straight from Source Cooperative
and writes the centroids locally:

```sql
INSTALL spatial; LOAD spatial;
INSTALL httpfs;  LOAD httpfs;
CREATE SECRET (
  TYPE S3, PROVIDER CONFIG,
  ENDPOINT 'data.source.coop', URL_STYLE 'path',
  REGION 'us-west-2', USE_SSL true
);
COPY (
  SELECT ST_AsWKB(ST_Centroid(geometry)) AS geometry
  FROM read_parquet('s3://hdx/microsoft-open-buildings/parquet/RegionName=UnitedStates/**/*.parquet')
) TO 'data/microsoft-buildings_point.parquet' (FORMAT PARQUET);
```

The WKB decoder (`decode_wkb_point`) only reads bytes 5..21 and assumes a
Point geometry — make sure the centroid step actually produces points, not
multipoints.

## Expected output

```
<N> points joined in ~4.3s (~30M pts/s)
<top 5 ZIPs by building count>
```

on a recent multi-core box with both files on local SSD.
