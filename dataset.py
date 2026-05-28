"""
dataset.py
----------------------------
Extrae valores de píxel de las bandas Sentinel-2 en las localizaciones
definidas por los polígonos de entrenamiento (entrenamiento.shp).

Genera un archivo TSV con columnas:
  Latitude, Longitude, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, label

USO:
  1. Ajustar BAND_FILES con las rutas a los .tif
  2. Ajustar CLASS_MAP con los nombres de las clases según el macroclass_id
  3. Ejecuta: python dataset.py
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURACIÓN — ajusta estas rutas y nombres
# ─────────────────────────────────────────────

SHAPEFILE = "entrenamiento.shp"   # ruta a tu shapefile de training

# Opción A: bandas separadas — pon la ruta a cada .tif
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

# Opción B: si tienes un stack multibanda, descomenta esto y comenta BAND_FILES
# STACK_FILE = "pre_event_stack.tif"   # bandas en orden: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12
# BAND_ORDER = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]

# Mapeo de macroclass_id → nombre de clase
# Ajusta según los ROIs que dibujaste en QGIS/SCP
CLASS_MAP = {
    1: "Water",
    2: "Built-up",
    3: "Vegetation",
    4: "Soil",
}

OUTPUT_TSV = "training_dataset.tsv"

# ─────────────────────────────────────────────
# EXTRACCIÓN
# ─────────────────────────────────────────────

def extract_pixels_single_bands(gdf, band_files, class_map):
    """Extrae píxeles con archivos de banda separados."""
    records = []

    for _, row in gdf.iterrows():
        class_id = row["macroclass"]
        label = class_map.get(class_id, f"class_{class_id}")
        geom = [mapping(row.geometry)]

        # Extraer valores de cada banda para este polígono
        band_values = {}
        coords = None

        for band_name, tif_path in band_files.items():
            with rasterio.open(tif_path) as src:
                try:
                    out_image, out_transform = mask(src, geom, crop=True)
                    data = out_image[0]  # primera banda del archivo
                    nodata = src.nodata if src.nodata is not None else 0

                    # Obtener coordenadas de píxeles válidos (primer banda)
                    if coords is None:
                        rows_idx, cols_idx = np.where(data != nodata)
                        xs, ys = rasterio.transform.xy(
                            out_transform, rows_idx, cols_idx, offset="center"
                        )
                        # Convertir a lat/lon si el CRS no es geográfico
                        from pyproj import Transformer
                        crs = src.crs.to_epsg()
                        if crs != 4326:
                            transformer = Transformer.from_crs(
                                f"EPSG:{crs}", "EPSG:4326", always_xy=True
                            )
                            lons, lats = transformer.transform(xs, ys)
                        else:
                            lons, lats = xs, ys
                        coords = list(zip(lats, lons))

                    band_values[band_name] = data[rows_idx, cols_idx].astype(float)
                except Exception as e:
                    print(f"  Warning: polígono {row['roi_id']} banda {band_name}: {e}")
                    continue

        if coords and band_values:
            n = len(coords)
            for i in range(n):
                rec = {
                    "Latitude":  coords[i][0],
                    "Longitude": coords[i][1],
                }
                for band_name in band_files.keys():
                    rec[band_name] = band_values[band_name][i] if i < len(band_values.get(band_name, [])) else np.nan
                rec["label"] = label
                records.append(rec)

    return records


def extract_pixels_stack(gdf, stack_file, band_order, class_map):
    """Extrae píxeles desde un stack multibanda."""
    records = []

    with rasterio.open(stack_file) as src:
        crs_epsg = src.crs.to_epsg()
        from pyproj import Transformer
        transformer = None
        if crs_epsg != 4326:
            transformer = Transformer.from_crs(f"EPSG:{crs_epsg}", "EPSG:4326", always_xy=True)

        for _, row in gdf.iterrows():
            class_id = row["macroclass"]
            label = class_map.get(class_id, f"class_{class_id}")
            geom = [mapping(row.geometry)]

            try:
                out_image, out_transform = mask(src, geom, crop=True)
                # out_image shape: (n_bands, height, width)
                valid_mask = out_image[0] != (src.nodata if src.nodata else 0)
                rows_idx, cols_idx = np.where(valid_mask)
                xs, ys = rasterio.transform.xy(out_transform, rows_idx, cols_idx, offset="center")

                if transformer:
                    lons, lats = transformer.transform(xs, ys)
                else:
                    lons, lats = xs, ys

                for i in range(len(lats)):
                    rec = {"Latitude": lats[i], "Longitude": lons[i]}
                    for b_idx, b_name in enumerate(band_order):
                        rec[b_name] = float(out_image[b_idx, rows_idx[i], cols_idx[i]])
                    rec["label"] = label
                    records.append(rec)

            except Exception as e:
                print(f"  Warning: polígono {row['roi_id']}: {e}")

    return records


def main():
    print("Leyendo shapefile de training...")
    gdf = gpd.read_file(SHAPEFILE)
    print(f"  {len(gdf)} polígonos cargados")
    print(f"  Clases encontradas: {gdf['macroclass'].value_counts().to_dict()}")

    # Elegir modo según configuración
    use_stack = "STACK_FILE" in dir() or False  # cambia a True si usas stack
    
    if use_stack:
        print(f"Modo: stack multibanda ({STACK_FILE})")
        records = extract_pixels_stack(gdf, STACK_FILE, BAND_ORDER, CLASS_MAP)
    else:
        print("Modo: bandas separadas")
        records = extract_pixels_single_bands(gdf, BAND_FILES, CLASS_MAP)

    df = pd.DataFrame(records)
    cols = ["Latitude", "Longitude", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "label"]
    df = df[cols]

    print(f"\nResumen del dataset:")
    print(f"  Total píxeles: {len(df)}")
    print(f"  Por clase:\n{df['label'].value_counts().to_string()}")
    print(f"\nPrimeras filas:")
    print(df.head())

    # Balancear el dataset
    min_pixels = df['label'].value_counts().min()
    print(f"\nMuestreando {min_pixels} píxeles por clase...")
    df_balanced = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=min_pixels, random_state=42)
    ).reset_index(drop=True)
    print(f"Dataset balanceado:\n{df_balanced['label'].value_counts().to_string()}")

    df_balanced.to_csv(OUTPUT_TSV, sep="\t", index=False)  #guarda el balanceado
    print(f"\nDataset guardado en: {OUTPUT_TSV}")


if __name__ == "__main__":
    main()