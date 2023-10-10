import numpy as np
import festim as F


def test_different_volume_ids():
    my_model = F.HydrogenTransportProblem()

    sub_dom_1 = np.linspace(0, 1e-04, num=3)
    sub_dom_2 = np.linspace(1e-04, 2e-04, num=4)
    sub_dom_3 = np.linspace(2e-04, 3e-04, num=5)
    my_model.mesh = F.Mesh1D(
        np.unique(np.concatenate([sub_dom_1, sub_dom_2, sub_dom_3]))
    )

    vol_subdomains_ids = [2, 16, 7]
    vol_subdomain_1 = F.VolumeSubdomain1D(
        id=vol_subdomains_ids[0], borders=[sub_dom_1[0], sub_dom_1[-1]]
    )
    vol_subdomain_2 = F.VolumeSubdomain1D(
        id=vol_subdomains_ids[1], borders=[sub_dom_2[0], sub_dom_2[-1]]
    )
    vol_subdomain_3 = F.VolumeSubdomain1D(
        id=vol_subdomains_ids[2], borders=[sub_dom_3[0], sub_dom_3[-1]]
    )
    my_model.subdomains = [vol_subdomain_1, vol_subdomain_2, vol_subdomain_3]

    my_model.define_markers_and_measures()

    for vol_id in vol_subdomains_ids:
        assert vol_id in np.array(my_model.volume_meshtags.values)


if __name__ == "__main__":
    test_different_volume_ids()
