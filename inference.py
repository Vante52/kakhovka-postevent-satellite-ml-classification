"""
inference.py
─────────────────────────────────────────────────────────────────────────────
Full-scene inference using the best-performing SVM model.
The prediction is clipped to the Region of Interest (ROI) defined by the
bounding box of the training polygons (entrenamiento.shp).

Workflow:
  1. Read ROI extent from entrenamiento.shp.
  2. Train SVM on the complete labeled dataset (all 8108 samples).
  3. Read all Sentinel-2 band rasters, clipped to the ROI window.
  4. Predict land cover class for every valid pixel inside the ROI.
  5. Write the result as a georeferenced GeoTIFF (ROI extent only).
  6. Print area statistics per class.
  7. Save a thematic map preview.

Output:
  predicted_landcover.tif  — single-band categorical raster (ROI only)
                             1 = Built-up | 2 = Soil | 3 = Vegetation | 4 = Water

Usage:
  python inference.py
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATASET_TSV  = "training_dataset.tsv"
SHAPEFILE    = "entrenamiento.shp"     # defines the ROI extent

BAND_FILES = {
    "B2":  "B2.tif",
    "B3":  "B3.tif",
    "B4":  "B4.tif",
    "B5":  "B5.tif",
    "B6":  "B6.tif",
    "B7":  "B7.tif",
    "B8":  "B8.tif",
    "B8A": "B8A.tif",
    "B11": "B11.tif",
    "B12": "B12.tif",
}
BAND_COLS = list(BAND_FILES.keys())

CLASS_CODES = {
    "Built-up":   1,
    "Soil":       2,
    "Vegetation": 3,
    "Water":      4,
}
CLASS_COLORS = {
    1: "#d9534f",   # Built-up   — red
    2: "#c8a96e",   # Soil       — tan
    3: "#5cb85c",   # Vegetation — green
    4: "#5bc0de",   # Water      — blue
}

OUTPUT_TIFF = "predicted_landcover.tif"
OUTPUT_MAP  = "docs/figures/thematic_map.png"

# ─────────────────────────────────────────────
# 1. READ ROI EXTENT FROM SHAPEFILE
# ─────────────────────────────────────────────

print("Reading ROI extent from shapefile...")
gdf = gpd.read_file(SHAPEFILE)
minx, miny, maxx, maxy = gdf.total_bounds
print(f"  ROI bounds (EPSG:{gdf.crs.to_epsg()}): "
      f"X [{minx:.0f} – {maxx:.0f}]  Y [{miny:.0f} – {maxy:.0f}]")
print(f"  Approx. size: {(maxx-minx)/1000:.1f} x {(maxy-miny)/1000:.1f} km")

# ─────────────────────────────────────────────
# 2. TRAIN SVM ON FULL DATASET
# ─────────────────────────────────────────────

print("\nLoading training dataset...")
df = pd.read_csv(DATASET_TSV, sep="\t")
print(f"  {len(df)} samples — classes: {df['label'].value_counts().to_dict()}")

X_all = df[BAND_COLS].values
y_all = df["label"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y_all)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

print("Training SVM on full dataset (this may take a minute)...")
svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
svm.fit(X_scaled, y_encoded)
print("  Done.")

# ─────────────────────────────────────────────
# 3. LOAD RASTER STACK CLIPPED TO ROI
# ─────────────────────────────────────────────

print("\nReading raster bands clipped to ROI...")

with rasterio.open(BAND_FILES["B2"]) as ref:
    # Compute the pixel window corresponding to the ROI bounding box
    roi_window = from_bounds(minx, miny, maxx, maxy, ref.transform)
    roi_window = roi_window.round_offsets().round_lengths()

    roi_transform = ref.window_transform(roi_window)
    roi_height    = int(roi_window.height)
    roi_width     = int(roi_window.width)
    profile       = ref.profile.copy()
    nodata_val    = ref.nodata if ref.nodata is not None else 0

print(f"  ROI window: {roi_width} x {roi_height} pixels")

band_stack = np.zeros((len(BAND_COLS), roi_height, roi_width), dtype=np.float32)

for i, band_name in enumerate(BAND_COLS):
    with rasterio.open(BAND_FILES[band_name]) as src:
        band_stack[i] = src.read(1, window=roi_window).astype(np.float32)

print("  Band stack ready.")

# ─────────────────────────────────────────────
# 4. PREDICT IN ROW CHUNKS (memory-safe)
# ─────────────────────────────────────────────

CHUNK_ROWS = 200

out_array = np.zeros((roi_height, roi_width), dtype=np.uint8)

print(f"\nRunning inference in chunks of {CHUNK_ROWS} rows...")

for row_start in range(0, roi_height, CHUNK_ROWS):
    row_end  = min(row_start + CHUNK_ROWS, roi_height)
    chunk    = band_stack[:, row_start:row_end, :]
    chunk_h  = row_end - row_start

    X_chunk = chunk.reshape(len(BAND_COLS), chunk_h * roi_width).T

    valid = ~np.all(X_chunk == nodata_val, axis=1)

    if valid.any():
        X_valid_scaled = scaler.transform(X_chunk[valid])
        y_enc    = svm.predict(X_valid_scaled)
        y_labels = le.inverse_transform(y_enc)
        y_codes  = np.array([CLASS_CODES[l] for l in y_labels], dtype=np.uint8)

        out_chunk = np.zeros(chunk_h * roi_width, dtype=np.uint8)
        out_chunk[valid] = y_codes
        out_array[row_start:row_end, :] = out_chunk.reshape(chunk_h, roi_width)

    if (row_start // CHUNK_ROWS) % 5 == 0:
        pct = row_end / roi_height * 100
        print(f"  {pct:.0f}%  (rows {row_start}-{row_end})")

print("  Inference complete.")

# ─────────────────────────────────────────────
# 5. WRITE GEOTIFF (ROI ONLY)
# ─────────────────────────────────────────────

profile.update(
    dtype=rasterio.uint8,
    count=1,
    nodata=0,
    compress="lzw",
    width=roi_width,
    height=roi_height,
    transform=roi_transform,
)
profile.pop("photometric", None)

with rasterio.open(OUTPUT_TIFF, "w", **profile) as dst:
    dst.write(out_array, 1)

print(f"\nGeoTIFF saved: {OUTPUT_TIFF}")

# ─────────────────────────────────────────────
# 6. AREA STATISTICS
# ─────────────────────────────────────────────

pixel_area_ha  = abs(roi_transform.a * roi_transform.e) / 10_000
pixel_area_km2 = abs(roi_transform.a * roi_transform.e) / 1_000_000
code_to_name   = {v: k for k, v in CLASS_CODES.items()}

print("\n== Area per land cover class (ROI only) ==============")
print(f"  Pixel: {abs(roi_transform.a):.1f} m -> {pixel_area_ha:.4f} ha/pixel")
print(f"{'Class':<15} {'Pixels':>10} {'Area (ha)':>12} {'Area (km2)':>12}")
print("-" * 55)
for code in sorted(CLASS_CODES.values()):
    count = int(np.sum(out_array == code))
    print(f"  {code_to_name[code]:<13} {count:>10,} {count*pixel_area_ha:>12.1f} {count*pixel_area_km2:>12.3f}")
total = int(np.sum(out_array > 0))
print("-" * 55)
print(f"  {'TOTAL':<13} {total:>10,} {total*pixel_area_ha:>12.1f} {total*pixel_area_km2:>12.3f}")

# ─────────────────────────────────────────────
# 7. THEMATIC MAP PREVIEW
# ─────────────────────────────────────────────

print(f"\nGenerating thematic map -> {OUTPUT_MAP}")

rgb = np.zeros((roi_height, roi_width, 3), dtype=np.uint8)
for code, hx in CLASS_COLORS.items():
    r, g, b = int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16)
    rgb[out_array == code] = [r, g, b]

fig, ax = plt.subplots(figsize=(12, 7))
ax.imshow(rgb)
ax.set_title("Predicted Land Cover - Kakhovka Post-Event ROI (SVM)", fontsize=13)
ax.axis("off")
patches = [mpatches.Patch(color=CLASS_COLORS[c], label=code_to_name[c])
           for c in sorted(CLASS_CODES.values())]
ax.legend(handles=patches, loc="lower right", fontsize=10, framealpha=0.85)
plt.tight_layout()
plt.savefig(OUTPUT_MAP, dpi=150, bbox_inches="tight")
plt.show()
print(f"Thematic map saved: {OUTPUT_MAP}")
