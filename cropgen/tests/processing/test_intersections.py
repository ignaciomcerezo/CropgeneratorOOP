from cropgen.tests.tests_helper import load_particular_annotation


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

    ann103 = load_particular_annotation(paths, 103, 1)

    # bloques [A][B][C] adyacentes, por otra parte [D][E]

    box_a = ann103.image_boxes["IiE7GGxUDC"]
    box_b = ann103.image_boxes["2xbI1Hl0SF"]
    box_c = ann103.image_boxes["55lKzt7x5K"]
    box_d = ann103.image_boxes["pUwiyxx5ef"]
    box_e = ann103.image_boxes["naknC3zYol"]

    assert box_a.polygon.intersects(box_b.polygon)
    assert not box_a.polygon.intersects(box_c.polygon)
    assert not box_a.polygon.intersects(box_d.polygon)
    assert not box_a.polygon.intersects(box_e.polygon)

    assert box_b.polygon.intersects(box_c.polygon)
    assert not box_b.polygon.intersects(box_d.polygon)
    assert not box_b.polygon.intersects(box_e.polygon)

    assert not box_c.polygon.intersects(box_d.polygon)
    assert not box_c.polygon.intersects(box_e.polygon)

    assert box_d.polygon.intersects(box_e.polygon)

    assert len(ann103.graph[box_e.id]) == 2
