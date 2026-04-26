#!/usr/bin/env bash
# Populates data/ with the two parquet files the example expects.
#
# Requires: curl, unzip, ogr2ogr (GDAL >= 3.5 built with Parquet),
#           duckdb (with spatial + httpfs extensions).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
mkdir -p "$DATA"

for t in curl unzip ogr2ogr duckdb; do
  command -v "$t" >/dev/null || { echo "missing tool: $t" >&2; exit 1; }
done

# 1) US ZIP codes — Census ZCTA5 2020 shapefile -> GeoParquet.
ZIP_OUT="$DATA/us-zip-codes.parquet"
if [ ! -f "$ZIP_OUT" ]; then
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  echo "[1/2] downloading ZCTA5 shapefile..."
  curl -fL --retry 3 -o "$TMP/zcta.zip" \
    https://www2.census.gov/geo/tiger/TIGER2020/ZCTA520/tl_2020_us_zcta520.zip
  unzip -q "$TMP/zcta.zip" -d "$TMP"
  echo "[1/2] converting to GeoParquet..."
  ogr2ogr -f Parquet "$ZIP_OUT" \
    -nlt MULTIPOLYGON \
    -sql "SELECT ZCTA5CE20 AS zipcode FROM tl_2020_us_zcta520" \
    "$TMP/tl_2020_us_zcta520.shp"
  rm -rf "$TMP"
  trap - EXIT
else
  echo "[1/2] $ZIP_OUT already exists, skipping."
fi

# 2) Microsoft building centroids.
# Default source is the HDX Microsoft Open Buildings USA partition on Source
# Cooperative (~141M buildings, ~12 GB streamed). DuckDB reads it directly via
# S3, computes centroids, and writes a local parquet of WKB Point bytes.
# Override with $BUILDINGS_SRC to point at a different glob (local or remote).
BLD_OUT="$DATA/microsoft-buildings_point.parquet"
BUILDINGS_SRC="${BUILDINGS_SRC:-s3://hdx/microsoft-open-buildings/parquet/RegionName=UnitedStates/**/*.parquet}"
if [ ! -f "$BLD_OUT" ]; then
  echo "[2/2] computing centroids from $BUILDINGS_SRC ..."
  echo "      (streaming ~141M buildings; this will take a while.)"
  duckdb -c "
    INSTALL spatial; LOAD spatial;
    INSTALL httpfs;  LOAD httpfs;
    CREATE SECRET (
      TYPE S3, PROVIDER CONFIG,
      ENDPOINT 'data.source.coop', URL_STYLE 'path',
      REGION 'us-west-2', USE_SSL true
    );
    COPY (
      SELECT ST_AsWKB(ST_Centroid(geometry)) AS geometry
      FROM read_parquet('$BUILDINGS_SRC')
    ) TO '$BLD_OUT' (FORMAT PARQUET);
  "
else
  echo "[2/2] $BLD_OUT already exists, skipping."
fi

echo
echo "data/:"
ls -lh "$DATA"
