from cropgen.tests.tests_helper import load_ann


def test_intersections_and_geometries(paths, lsi):

    ann = load_ann(paths, 4, 0, fake_image=True)

    boxid2line = {line.box_id: line for line in ann.lines.values()}

    box_a = boxid2line["_3qKT12Lkm"]
    box_b = boxid2line["9bD_DAaJ9I"]
    box_c = boxid2line["R0cpzJhaF8"]

    assert len(box_a.polygon.exterior.coords[:]) > 4
    assert len(box_b.polygon.exterior.coords[:]) > 4
    assert len(box_c.polygon.exterior.coords[:]) > 4

    assert not box_a.true_rectangle
    assert not box_b.true_rectangle
    assert not box_c.true_rectangle

    assert box_a.polygon.intersects(box_b.polygon)
    assert box_b.polygon.intersects(box_c.polygon)
    assert not box_c.polygon.intersects(box_a.polygon)

    assert len(load_ann(paths, 341, 0, fake_image=True).paragraphs) == 1
    assert len(load_ann(paths, 342, 0, fake_image=True).paragraphs) == 1
    assert len(load_ann(paths, 343, 0, fake_image=True).paragraphs) == 2
    assert len(load_ann(paths, 344, 1, fake_image=True).paragraphs) == 2

    ann = load_ann(paths, 103, 1)

    # bloques [A][B][C] adyacentes, por otra parte [D][E]

    boxid2line = {line.box_id: line for line in ann.lines.values()}

    box_a = boxid2line["IiE7GGxUDC"]
    box_b = boxid2line["2xbI1Hl0SF"]
    box_c = boxid2line["55lKzt7x5K"]
    box_d = boxid2line["pUwiyxx5ef"]
    box_e = boxid2line["naknC3zYol"]

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

    assert len(ann.graph[box_e.id]) == 2
