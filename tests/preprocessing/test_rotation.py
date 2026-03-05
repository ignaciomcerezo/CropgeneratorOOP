from tests.tests_helper import load_particular_annotation

Ann = load_particular_annotation(344, 1)


assert len(Ann.paragraphs) == 2

img, txt, sindex = Ann.paragraphs[0].cluster_reading_order()
assert (
    txt
    == r"""voir à part le cas $\nu=1$ : normalisateur du sous-groupe engendré par $\prod [x_\alpha,y_\alpha]$, dans le groupe libre engendré par les générateurs $x_\alpha$, $y_\alpha$,..."""
)

img, txt, sindex = Ann.paragraphs[1].cluster_reading_order()
assert (
    txt
    == r"""On suppose dorénavant qu'on est dans la cas anabélien, ou du moins on exclut $g=0,\nu\leq 2$"""
)
#
# print(Ann.paragraphs[0].avg_rotation)
# print(Ann.paragraphs[1].avg_rotation)
# list(Ann.image_boxes.values())[1].crop.show()
# par = Ann.paragraphs[1]
# from shapely import Point, affinity, Polygon, coverage_union_all
# import numpy as np
#
# def visually_correct_polygon(pol: Polygon):
#     return Polygon([(x, y) for (x,y) in pol.exterior.coords[:]])
#
# def transform_geometry(geom, theta_degrees, centroid: tuple[float, float] = (0,0)):
#     theta_rad = np.radians(theta_degrees)
#
#     a = np.cos(theta_rad)
#     b = -np.sin(theta_rad)
#     d = np.sin(theta_rad)
#     e = np.cos(theta_rad)
#
#     matrix = [a, b, d, e, -centroid[0], -centroid[1]]
#     return affinity.affine_transform(geom, matrix)
#
# points = []
#
# for i in range(len(par.image_boxes)):
#     tf = par.text_fragments[i]
#     ib = par.image_boxes[i]
#
#     points.append( Point(ib.centroid()))
#
#
#
#     print(tf, "///", ib.corrected_centroid)
#
# from shapely import MultiPoint
#
# vct_polygons: list[Polygon] = []
# new_corr_centroids = []
# bounding_rectangles = []
#
# display(MultiPoint(points))
# for pol in (box.polygon for box in par.image_boxes):
#     vct_polygons.append(transform_geometry(pol, par.avg_rotation, par.centroid))
#     new_corr_centroids.append(vct_polygons[-1].centroid)
#     bounding_rectangles.append(vct_polygons[-1].minimum_rotated_rectangle)
#
# display(coverage_union_all(vct_polygons))
# display(MultiPoint(new_corr_centroids))
# display(coverage_union_all(bounding_rectangles))
#
# sorted(vct_polygons, key = lambda x: min([y[1] for y in x.exterior.coords[:]]))[-1]
