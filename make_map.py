"""Generate area stats and thematic map from the existing predicted_landcover.tif."""
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_TIFF = "predicted_landcover.tif"
OUTPUT_MAP  = "docs/figures/thematic_map.png"

CLASS_CODES  = {"Built-up": 1, "Soil": 2, "Vegetation": 3, "Water": 4}
CLASS_COLORS = {1: "#d9534f", 2: "#c8a96e", 3: "#5cb85c", 4: "#5bc0de"}
code_to_name = {v: k for k, v in CLASS_CODES.items()}

with rasterio.open(OUTPUT_TIFF) as src:
    out_array = src.read(1)
    t = src.transform

pixel_area_ha  = abs(t.a * t.e) / 10_000
pixel_area_km2 = abs(t.a * t.e) / 1_000_000

print("== Area per land cover class ==")
print(f"  Pixel: {abs(t.a):.1f} m -> {pixel_area_ha:.4f} ha")
print(f"{'Class':<15} {'Pixels':>10} {'Area (ha)':>12} {'Area (km2)':>12}")
print("-" * 55)
for code in sorted(CLASS_CODES.values()):
    count = int(np.sum(out_array == code))
    print(f"  {code_to_name[code]:<13} {count:>10,} {count*pixel_area_ha:>12.1f} {count*pixel_area_km2:>12.3f}")
total = int(np.sum(out_array > 0))
print("-" * 55)
print(f"  {'TOTAL':<13} {total:>10,} {total*pixel_area_ha:>12.1f} {total*pixel_area_km2:>12.3f}")

# Thematic map
h, w = out_array.shape
rgb = np.zeros((h, w, 3), dtype=np.uint8)
for code, hx in CLASS_COLORS.items():
    r, g, b = int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16)
    rgb[out_array == code] = [r, g, b]

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(rgb)
ax.set_title("Predicted Land Cover - Kakhovka Post-Event (SVM)", fontsize=13)
ax.axis("off")
patches = [mpatches.Patch(color=CLASS_COLORS[c], label=code_to_name[c])
           for c in sorted(CLASS_CODES.values())]
ax.legend(handles=patches, loc="lower right", fontsize=10, framealpha=0.85)
plt.tight_layout()
plt.savefig(OUTPUT_MAP, dpi=150, bbox_inches="tight")
print(f"\nThematic map saved: {OUTPUT_MAP}")
