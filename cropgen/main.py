from cropgen.shared.PathBundle import PathBundle
from cropgen.external_interfaces.LabelStudioInterface import LabelStudioInterface
from cropgen.external_interfaces.OracleBucketInterface import OracleBucketInterface
from cropgen.splitter.crops_interface.PairsDataInterface import PairsDataInterface
from dotenv import load_dotenv
from cropgen.processing.parallel.augment_data_parallel import augment_data_parallel

load_dotenv()
paths = PathBundle()
obi = OracleBucketInterface.from_env(paths)
obi.update()
has_updated = LabelStudioInterface.update_conditional(paths)
lsi = LabelStudioInterface(paths)
if has_updated:
    lsi.save_simplified_export()

pdi = PairsDataInterface(paths)
