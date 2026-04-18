from dotenv import load_dotenv

from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
from cropgen.external_interfaces.OracleBucketInterface import OracleBucketInterface
from cropgen.processing.parallel.augment_data_parallel import augment_data_parallel
from cropgen.shared.PathBundle import PathBundle


def generate(
    paths: PathBundle | None = None,
    obi: OracleBucketInterface | None = None,
    lsi: LabelStudioInterface | None = None,
):
    # TODO: recuerda que los fragmentos que se eliminan del grafo tienen starting_index = -1
    load_dotenv()
    paths: PathBundle = PathBundle() if paths is None else paths

    obi: OracleBucketInterface = (
        OracleBucketInterface.from_env(paths) if obi is None else obi
    )
    obi.update()
    has_updated = LabelStudioInterface.update_conditional(paths)
    lsi: LabelStudioInterface = LabelStudioInterface(paths) if lsi is None else lsi
    if has_updated:
        lsi.save_simplified_export()

    augment_data_parallel(
        paths,
        tasks_only=None,
        orders_to_consider=[0],
        generate_full_pages=True,
        generate_paragraphs=False,
        num_processes=6,
        lsi=lsi,
    )


if __name__ == "__main__":
    generate()
