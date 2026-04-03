from cropgen.shared.PathBundle import PathBundle
from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
from cropgen.external_interfaces.OracleBucketInterface import OracleBucketInterface
from cropgen.splitter.crops_interface.PairsDataInterface import PairsDataInterface
from dotenv import load_dotenv
from cropgen.processing.parallel.augment_data_parallel import augment_data_parallel
from cropgen.splitter.generation.get_dataset import get_datasets

load_dotenv()
paths = PathBundle()
obi = OracleBucketInterface.from_env(paths)
obi.update()
has_updated = LabelStudioInterface.update_conditional(paths)
lsi = LabelStudioInterface(paths)
if has_updated:
    lsi.save_simplified_export()

pdi = PairsDataInterface(paths)


def main():
    # TODO: recuerda que los fragmentos que se eliminan del grafo tienen starting_index = -1
    augment_data_parallel(
        paths,
        tasks_only=None,
        orders_to_consider=[1, 2, 3],
        generate_full_pages=True,
        generate_paragraphs=True,
        num_processes=4,
        lsi=lsi,
    )


kappa = True

if kappa:
    kappa = False
    main()
