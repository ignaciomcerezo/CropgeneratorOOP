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


def main():
    augment_data_parallel(paths, [1], True, True, tasks_only=[15], lsi=lsi)


if __name__ == "__main__":
    main()
