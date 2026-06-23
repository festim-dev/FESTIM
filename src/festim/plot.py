from pathlib import Path

from festim.species import Species

DEFAULT_TITLE_FONT_SIZE = 12


def _normalize_fields(field: Species | list[Species]) -> list[Species]:
    if isinstance(field, Species):
        return [field]
    if isinstance(field, list) and all(isinstance(f, Species) for f in field):
        return field
    raise TypeError("field must be of type festim.Species or a list of festim.Species")


def _get_solution(field: Species, subdomain=None):
    if subdomain is None:
        if field.post_processing_solution is None:
            raise ValueError(
                f"Species {field.name} has no post_processing_solution to plot."
            )
        return field.post_processing_solution

    if not field.subdomain_to_post_processing_solution:
        raise ValueError(
            f"Species {field.name} has no subdomain post-processing solutions."
        )
    if subdomain not in field.subdomain_to_post_processing_solution:
        raise ValueError(
            f"Species {field.name} has no post-processing solution on subdomain "
            f"{subdomain}."
        )
    return field.subdomain_to_post_processing_solution[subdomain]


def _make_ugrid(solution, pyvista_module):
    from dolfinx import plot as dolfinx_plot

    topology, cell_types, geometry = dolfinx_plot.vtk_mesh(solution.function_space)
    u_grid = pyvista_module.UnstructuredGrid(topology, cell_types, geometry)
    u_grid.point_data["c"] = solution.x.array.real
    u_grid.set_active_scalars("c")
    return u_grid


def plot(
    field: Species | list[Species],
    subdomain=None,
    filename: str | Path | None = None,
    show_edges: bool = False,
    **kwargs,
):
    """
    Plot one or several species fields with pyvista.

    Args:
        field: one species or a list of species.
        subdomain: optional volume subdomain used in mixed-domain problems.
        filename: optional output image path. If provided, a screenshot is saved.
        show_edges: whether to show mesh edges.
        **kwargs: additional arguments forwarded to ``pyvista.Plotter.add_mesh``.
    """
    try:
        import pyvista
    except ImportError as import_error:
        raise ImportError(
            "pyvista is required for plotting. Install it with `pip install pyvista`."
        ) from import_error

    fields = _normalize_fields(field)
    shape = (1, len(fields)) if len(fields) > 1 else None
    plotter = pyvista.Plotter(shape=shape)

    for i, spe in enumerate(fields):
        if len(fields) > 1:
            plotter.subplot(0, i)

        solution = _get_solution(spe, subdomain=subdomain)
        u_grid = _make_ugrid(solution, pyvista_module=pyvista)
        plotter.add_mesh(u_grid, show_edges=show_edges, **kwargs)
        plotter.view_xy()
        if spe.name:
            plotter.add_text(spe.name, font_size=DEFAULT_TITLE_FONT_SIZE)

    if filename is not None:
        plotter.screenshot(str(filename))
    elif not pyvista.OFF_SCREEN:
        plotter.show()

    return plotter
