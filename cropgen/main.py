from shared.PathBundle import PathBundle
from external_interfaces.LabelStudioInterface import LabelStudioInterface
from external_interfaces.OracleBucketInterface import OracleBucketInterface
from dotenv import load_dotenv

load_dotenv()
paths = PathBundle()
OracleBucketInterface.from_env(paths).update()
LabelStudioInterface.update_conditional(paths)
lsi = LabelStudioInterface(paths)
lsi.save_simplified_export()

# TODO: check if the names that are being set @crops.xlsx are correct (they should match the name generation) - otherwise, check other related things
#
# def main():
#     augment_data_parallel(paths, [1], True, True, tasks_only=[15], lsi=lsi)
#
#
# if __name__ == "__main__":
#     main()
