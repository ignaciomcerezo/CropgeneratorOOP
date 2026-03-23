import os

from processing.parallel.augment_data_parallel import augment_data_parallel
from shared.PathBundle import PathBundle
from external_interfaces.LabelStudioInterface import LabelStudioInterface
from external_interfaces.OracleBucketInterface import OracleBucketInterface
import shutil
from dotenv import load_dotenv

load_dotenv()

paths = PathBundle()
OracleBucketInterface.from_env(paths).update()
LabelStudioInterface.update_conditional(paths)
lsi = LabelStudioInterface(paths)
lsi.save_simplified_export()
