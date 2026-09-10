
from cropgen.loading.external_interfaces.external_interface import ExternalInterface
from cropgen.loading.page_loader import load_pages
from cropgen.ocr_units.ocr_page import OCRPage
from cropgen.shared.path_bundle import PathBundle


def setup(
    paths: PathBundle,
    external_interfaces: list[ExternalInterface],
) -> list[OCRPage]:
    """
    Downloads all files needed to instanciate the dataset given some external interfaces and a path to store them.
    """
    parts = set()
    for i, external_interface in enumerate(external_interfaces):
        pm = external_interface.parts_managed()
        if pm.intersection(parts):
            for other_external_interface in external_interfaces[:i]:
                other_pm = other_external_interface.parts_managed()
                if other_pm.intersection(pm):
                    break
            raise ValueError(
                f"External interface conflict: {other_external_interface} and {external_interface} manage the same parts: {pm.intersection(other_pm)}"
            )
        pr = external_interface.parts_required()
        if not pr.issubset(parts):
            prev_msg = (
                f"(after {external_interfaces[:i]})"
                if i != 0
                else "(it is the first one)"
            )
            raise ValueError(
                f"External interface {external_interface} needs parts {pr} to setup, but "
                f"only {parts} is setup when it is called {prev_msg}. Try reordering the external interfaces to solve this, "
                "or perhaps the combination you chose is just incompatible."
            )
        parts.update(pm)

    for external_interface in external_interfaces:
        external_interface.setup(paths)

    return load_pages(paths)
