from cropgen.tests.tests_helper import load_ann


def test_intersections_and_geometries(paths, lsi):

    ann = load_ann(paths, 4, 0, lsi=lsi, fake_image=True)
    box_a = ann.image_boxes["_3qKT12Lkm"]
    box_b = ann.image_boxes["9bD_DAaJ9I"]
    box_c = ann.image_boxes["R0cpzJhaF8"]

    assert len(box_a.polygon.exterior.coords[:]) > 4
    assert len(box_b.polygon.exterior.coords[:]) > 4
    assert len(box_c.polygon.exterior.coords[:]) > 4

    assert not box_a.true_rectangle
    assert not box_b.true_rectangle
    assert not box_c.true_rectangle

    assert box_a.polygon.intersects(box_b.polygon)
    assert box_b.polygon.intersects(box_c.polygon)
    assert not box_c.polygon.intersects(box_a.polygon)

    assert len(load_ann(paths, 341, 0, lsi=lsi, fake_image=True).paragraphs) == 1
    assert len(load_ann(paths, 342, 0, lsi=lsi, fake_image=True).paragraphs) == 1
    assert len(load_ann(paths, 343, 0, lsi=lsi, fake_image=True).paragraphs) == 2
    assert len(load_ann(paths, 344, 1, lsi=lsi, fake_image=True).paragraphs) == 2

    ann103 = load_ann(paths, 103, 1)

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
