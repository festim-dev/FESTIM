"""Recover the experimental points of Fig. 4 from the published PDF.

Fig. 4 is the whole mixed-gas campaign: four stacked panels, one per membrane
temperature, each carrying four total pressures distinguished by colour and marker.
That is 16 conditions against Fig. 5's two, so it is the dataset worth fitting to.

The extraction works the same way as ``digitise_fig5.py`` -- threshold on hue, fill the
open markers, then keep only what holds a larger inscribed disk than the curves can --
with two additions the four-panel layout forces:

* **each panel is calibrated separately.** The four frames are not pixel-identical (they
  span 937 to 948 px for the same 0-1200 sccm), and calibrating them all from one frame
  puts the top panel out by ~18 sccm. Panel frames are found per panel instead.
* **the grey series has to be kept away from black text.** Antialiased black glyphs pass
  any "unsaturated mid-grey" test, so the panel annotations came through as data points.
  Dilating the black mask and subtracting it removes them; the triangles have no black
  near them and survive.

There is a check on all this built in. The campaign stepped the feed in 100 sccm
intervals, so every recovered point should land on a multiple of 100 -- and does, within
3.4 sccm worst case. ``main()`` prints that residual and then snaps the abscissa, which
removes the digitisation error there entirely.

Points are lost only where markers of different series overlap, mostly at the lowest
feed rates where all four pressures converge. 102 of about 120 survive.

Run it from this directory with the PDF alongside::

    python digitise_fig4.py            # rewrites fig4_digitised.csv
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
CSV = HERE / "fig4_digitised.csv"

FIGURE_PAGE = 4
DPI = 400

#: top and bottom frame rows of each panel, at 400 dpi, keyed by membrane temperature
PANELS = {
    723.15: (1631, 2011),
    673.15: (2064, 2443),
    623.15: (2495, 2871),
    573.15: (2923, 3307),
}
FEED_RANGE = 1200.0  # sccm, the full width of every panel
PERMEATED_RANGE = 12.0  # sccm, the full height of every panel

#: nothing above this in the 300 C panel is data -- it is the legend, which sits inside
LEGEND_FLOOR = 8.5

#: per series: the hue test, the distance-transform threshold, and the smallest core.
#: diamonds and triangles enclose less than squares, so they need a lower bar.
SERIES = {
    250e3: (lambda r, g, b, v, s: (b - r > 40) & (b - g > 20), 5.0, 18),
    190e3: (lambda r, g, b, v, s: (s < 25) & (v > 95) & (v < 205), 4.5, 12),
    150e3: (lambda r, g, b, v, s: (r - b > 60) & (g - b > 40), 4.0, 12),
    90e3: (lambda r, g, b, v, s: (g - r > 25) & (g - b > 30), 5.0, 18),
}
GREY = 190e3  # the one that needs protecting from black text


def render_page() -> np.ndarray:
    """Rasterises the page and returns its left column, where Fig. 4 sits."""
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
    return page[:, : page.shape[1] // 2]


def panel_frame(page: np.ndarray, top: int, bottom: int) -> tuple[float, float]:
    """The left and right frame columns of one panel.

    The left one is the column that is dark down the whole height of the panel. The
    right one is taken from the extent of the top frame row instead, because a marker
    sitting on the right frame can break the column test.
    """
    dark = page.sum(2) < 250
    column = dark[top:bottom, :].sum(0) / (bottom - top)
    return float(np.where(column > 0.75)[0].min()), float(np.where(dark[top])[0].max())


def extract(page: np.ndarray, temperature: float) -> dict[float, list]:
    """Every series of one panel, in data coordinates."""
    top, bottom = PANELS[temperature]
    left, right = panel_frame(page, top, bottom)
    panel = page[top:bottom, int(left) : int(right) + 1]

    red, green, blue = panel[:, :, 0], panel[:, :, 1], panel[:, :, 2]
    value, saturation = panel.mean(2), panel.max(2) - panel.min(2)
    near_black = ndimage.binary_dilation(value < 70, np.ones((13, 13)))

    found = {}
    for pressure, (is_series, threshold, min_core) in SERIES.items():
        mask = is_series(red, green, blue, value, saturation)
        if pressure == GREY:
            mask = mask & ~near_black
        mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, np.ones((5, 5))))
        labels, count = ndimage.label(ndimage.distance_transform_edt(mask) > threshold)

        points = []
        for index in range(1, count + 1):
            rows, columns = np.where(labels == index)
            if len(rows) < min_core:
                continue
            permeated = PERMEATED_RANGE * (1 - rows.mean() / (bottom - top))
            if temperature == 573.15 and permeated > LEGEND_FLOOR:
                continue
            points.append((FEED_RANGE * columns.mean() / (right - left), permeated))
        points.sort()

        merged: list[tuple[float, float]] = []
        for feed, permeated in points:
            if merged and feed - merged[-1][0] < 20:
                merged[-1] = (
                    (merged[-1][0] + feed) / 2,
                    (merged[-1][1] + permeated) / 2,
                )
            else:
                merged.append((feed, permeated))
        found[pressure] = merged
    return found


def main() -> None:
    page = render_page()
    rows, residuals = [], []
    for temperature in PANELS:
        for pressure, points in extract(page, temperature).items():
            for feed, permeated in points:
                snapped = round(feed / 100) * 100
                residuals.append(feed - snapped)
                rows.append((temperature, pressure, snapped, permeated))

    residuals = np.array(residuals)
    print(
        f"{len(rows)} points; the campaign stepped the feed by 100 sccm, and every"
        f" point lands within {np.abs(residuals).max():.1f} sccm of a multiple of it"
        f" (mean {residuals.mean():+.1f}). Snapping."
    )

    rows.sort()
    with CSV.open("w") as stream:
        stream.write(
            "# Experimental points digitised from Fig. 4 of Fuerst, Taylor & Shimada,\n"
            "# IEEE Trans. Plasma Sci. 52 (2024) 3925, doi:10.1109/TPS.2024.3356857.\n"
            "# Produced by digitise_fig4.py from the published PDF. 3.95% D2 in He.\n"
            "# Feed rates are snapped to the 100 sccm steps the campaign used; the\n"
            "# ordinate is good to about +/- 0.05 sccm, the size of a marker.\n"
            "# Points are missing where markers of different pressures overlap.\n"
            "temperature_K,total_pressure_Pa,feed_sccm,permeated_sccm\n"
        )
        for temperature, pressure, feed, permeated in rows:
            stream.write(f"{temperature},{pressure:.0f},{feed:.0f},{permeated:.3f}\n")
    print(f"wrote {CSV}")


if __name__ == "__main__":
    main()
