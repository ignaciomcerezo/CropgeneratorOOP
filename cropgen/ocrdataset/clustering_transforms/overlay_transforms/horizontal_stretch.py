from cropgen.ocrdataset.clustering_transforms.clustering_transform import (
    OverlayTransform,
)
from cropgen.shared.parameters import Parameter


class HorizontalStretch(OverlayTransform):
    def __init__(self, relative: Parameter):
        self.relative = relative

    # TODO: finish
