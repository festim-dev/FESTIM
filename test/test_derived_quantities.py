import os

import pytest

import festim as F

mobile_H = F.Species("H")
mobile_D = F.Species("D")
surf_1 = F.SurfaceSubdomain(id=1)
surf_2 = F.SurfaceSubdomain(id=2)
vol_1 = F.VolumeSubdomain(id=1, material="test")
vol_2 = F.VolumeSubdomain(id=2, material="test")
results = "test.csv"

surface_flux = F.SurfaceFlux(field=mobile_H, surface=surf_1, filename=results)
average_vol = F.AverageVolume(mobile_H, volume=vol_1, filename=results)
tot_surf = F.TotalSurface(mobile_D, surface=surf_2, filename=results)
tot_vol = F.TotalVolume(mobile_D, volume=vol_2, filename=results)
min_vol = F.MinimumVolume(mobile_H, volume=vol_1, filename=results)
max_vol = F.MaximumVolume(mobile_D, volume=vol_1, filename=results)
min_surface = F.MinimumSurface(mobile_D, surface=surf_1, filename=results)
max_surface = F.MaximumSurface(mobile_H, surface=surf_2, filename=results)
avg_surface = F.AverageSurface(mobile_D, surface=surf_1, filename=results)
avg_vol = F.AverageVolume(mobile_H, volume=vol_2, filename=results)
surf_quant = F.SurfaceQuantity(mobile_H, surface=surf_1, filename=results)
vol_quant = F.VolumeQuantity(mobile_H, volume=vol_1, filename=results)


@pytest.mark.parametrize(
    "quantity, expected_title",
    [
        (surface_flux, "H flux surface 1"),
        (average_vol, "Average H volume 1"),
        (tot_surf, "Total D surface 2"),
        (tot_vol, "Total D volume 2"),
        (min_vol, "Minimum H volume 1"),
        (max_vol, "Maximum D volume 1"),
        (min_surface, "Minimum D surface 1"),
        (max_surface, "Maximum H surface 2"),
        (avg_surface, "Average D surface 1"),
        (avg_vol, "Average H volume 2"),
    ],
)
def test_title(quantity, expected_title, tmp_path):
    quantity.filename = os.path.join(tmp_path, "test.csv")
    quantity.value = 1

    assert quantity.title == expected_title
