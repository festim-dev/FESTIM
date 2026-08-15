"""Recover the experimental points of Fig. 5 from the published PDF.

The paper reports the mixed-gas campaign only as figures, so comparing against it means
digitising. This does it by colour rather than by hand: the two series are drawn in
distinct blue and green, and the markers are fatter than any curve, so thresholding on
hue and then on the distance transform separates the data points from the model lines
they sit on.

Run it from this directory with the PDF alongside::

    python digitise_fig5.py            # rewrites fig5_digitised.csv

Accuracy is limited by the marker size, about +/- 0.05 sccm on the ordinate. The feed
rates come out within a few sccm of the round numbers the campaign actually used, which
is a fair estimate of the error.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).parent
PDF = next(HERE.glob("*Palladium-Silver*.pdf"), None)
CSV = HERE / "fig5_digitised.csv"

FIGURE_PAGE = 4  # Fig. 5 is top-right on page 4
DPI = 400

#: plot frame and axis calibration, in pixels of the rendered crop
PLOT_BOX = (321, 1181, 337, 1291)  # top, bottom, left, right
X_AXIS = (337.0, 1208.3, 0.0, 1000.0)  # px of two ticks, and their values
Y_AXIS = (1179.5, 320.5, 0.0, 25.0)
LEGEND_BOX = (321, 700, 337, 990)  # masked out: it repeats the marker colours

SERIES = {
    # name: (rgb test, distance-transform threshold in px)
    "450C_250kPa": (lambda r, g, b: (b > 110) & (b - r > 45) & (b - g > 25), 6.0),
    "300C_90kPa": (lambda r, g, b: (g > 90) & (g - r > 35) & (g - b > 35), 7.5),
}
CONDITIONS = {"450C_250kPa": (723.15, 250e3), "300C_90kPa": (573.15, 90e3)}


def render_figure() -> np.ndarray:
    """Rasterises the page and returns the top-right quadrant, where Fig. 5 sits."""
    if PDF is None:
        sys.exit(f"put the paper's PDF in {HERE} first")
    stem = HERE / "_page"
    subprocess.run(
        [
            "pdftoppm",
            "-r",
            str(DPI),
            "-f",
            str(FIGURE_PAGE),
            "-l",
            str(FIGURE_PAGE),
            "-png",
            str(PDF),
            str(stem),
        ],
        check=True,
    )
    rendered = next(HERE.glob("_page-*.png"))
    page = np.array(Image.open(rendered).convert("RGB")).astype(int)
    rendered.unlink()
    height, width = page.shape[:2]
    return page[: height // 2, width // 2 :]


def extract(image: np.ndarray, series: str) -> list[tuple[float, float]]:
    """The marker centroids of one series, in data coordinates."""
    is_series, threshold = SERIES[series]
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    top, bottom, left, right = PLOT_BOX
    inside = np.zeros(image.shape[:2], bool)
    inside[top:bottom, left:right] = True
    legend = np.zeros(image.shape[:2], bool)
    legend[LEGEND_BOX[0] : LEGEND_BOX[1], LEGEND_BOX[2] : LEGEND_BOX[3]] = True

    mask = is_series(red, green, blue) & inside & ~legend
    mask = ndimage.binary_closing(mask, np.ones((7, 7)))
    # the open squares are outlines: fill them so they are as solid as the circles
    mask = ndimage.binary_fill_holes(mask)
    # a marker holds a larger inscribed disk than any of the curves it lies on
    labels, count = ndimage.label(ndimage.distance_transform_edt(mask) > threshold)

    x_px0, x_px1, x_val0, x_val1 = X_AXIS
    y_px0, y_px1, y_val0, y_val1 = Y_AXIS
    points = []
    for index in range(1, count + 1):
        rows, columns = np.where(labels == index)
        if len(rows) < 25:
            continue
        points.append(
            (
                x_val0 + (columns.mean() - x_px0) * (x_val1 - x_val0) / (x_px1 - x_px0),
                y_val0 + (rows.mean() - y_px0) * (y_val1 - y_val0) / (y_px1 - y_px0),
            )
        )
    points.sort()

    merged: list[tuple[float, float]] = []
    for x, y in points:  # one marker can break into two cores
        if merged and x - merged[-1][0] < 25:
            merged[-1] = ((merged[-1][0] + x) / 2, (merged[-1][1] + y) / 2)
        else:
            merged.append((x, y))
    return merged


def main() -> None:
    image = render_figure()
    with CSV.open("w") as stream:
        stream.write(
            "# Experimental points digitised from Fig. 5 of Fuerst, Taylor & Shimada,\n"
            "# IEEE Trans. Plasma Sci. 52 (2024) 3925, doi:10.1109/TPS.2024.3356857.\n"
            "# Produced by digitise_fig5.py from the published PDF; feed rates land"
            " within\n# a few sccm of the round numbers the campaign used, which is the"
            " digitisation\n# error. 3.95% D2 in He.\n"
            "series,temperature_K,total_pressure_Pa,feed_sccm,permeated_sccm\n"
        )
        for series, (temperature, pressure) in CONDITIONS.items():
            points = extract(image, series)
            print(f"{series}: {len(points)} points")
            for feed, permeated in points:
                stream.write(
                    f"{series},{temperature},{pressure:.0f},{feed:.1f},{permeated:.3f}\n"
                )
    print(f"wrote {CSV}")


if __name__ == "__main__":
    main()
