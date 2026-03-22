from tests.tests_helper import load_particular_annotation


def test_intersections():
    ann5 = load_particular_annotation(5, 0)
    A = ann5.image_boxes["5LKgXZs7ij"]
    B = ann5.image_boxes["cf7ItU794h"]
    C = ann5.image_boxes["cXXa92pGsI"]

    assert len(A.polygon.exterior.coords[:]) > 4
    assert len(B.polygon.exterior.coords[:]) > 4
    assert len(C.polygon.exterior.coords[:]) > 4

    assert A.polygon.intersects(B.polygon)
    assert B.polygon.intersects(C.polygon)
    assert not C.polygon.intersects(A.polygon)

    ann5 = load_particular_annotation(344, 1)
    assert len(ann5.paragraphs) == 2

    assert len(load_particular_annotation(341, 0).paragraphs) == 1
    assert len(load_particular_annotation(342, 0).paragraphs) == 1
    assert len(load_particular_annotation(343, 0).paragraphs) == 2
