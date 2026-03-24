from src.cropgen.tests.tests_helper import load_particular_annotation


def test_intersections_and_geometries(paths):

    ann5 = load_particular_annotation(paths, 5, 0)
    box_a = ann5.image_boxes["5LKgXZs7ij"]
    box_b = ann5.image_boxes["cf7ItU794h"]
    box_c = ann5.image_boxes["cXXa92pGsI"]

    assert len(box_a.polygon.exterior.coords[:]) > 4
    assert len(box_b.polygon.exterior.coords[:]) > 4
    assert len(box_c.polygon.exterior.coords[:]) > 4

    assert not box_a.true_rectangle
    assert not box_b.true_rectangle
    assert not box_c.true_rectangle

    assert box_a.polygon.intersects(box_b.polygon)
    assert box_b.polygon.intersects(box_c.polygon)
    assert not box_c.polygon.intersects(box_a.polygon)

    assert len(load_particular_annotation(paths, 341, 0).paragraphs) == 1
    assert len(load_particular_annotation(paths, 342, 0).paragraphs) == 1
    assert len(load_particular_annotation(paths, 343, 0).paragraphs) == 2
    assert len(load_particular_annotation(paths, 344, 1).paragraphs) == 2
