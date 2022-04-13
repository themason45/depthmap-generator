import sys

import numpy as np
import shapely.geometry
import trimesh
from osgeo import ogr
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import triangulate
from trimesh import util
from trimesh.viewer import SceneViewer


def generate_bbox(x: int, y: int, r: int = 2000) -> (int, int, int, int):
    """
    NOTE: Use a metres based SRS (EPSG:3857)
    Generates a bounding box around a point:

    _________
    |       |
    |     r |
    |   .---|
    |   |   |
    |   | y |
    ----|----
      x

    :param x: The x coordinate
    :param y: The y coordinate
    :param r: The radius of the circle that we envelope
    :return: BBOX in the format: (minx,miny,maxx,maxy)
    """
    r /= 2
    return x - r, y - r, x + r, y + r


def generate_transform(arr):
    a = np.zeros((4, 4))
    a[:3, :3] = arr
    a[3, 3] = 1

    return a


def get_size(obj, seen=None):
    """Recursively finds size of objects"""

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])

    return size


def coord_string(x, y, z):
    s = (",".join(str(v) for v in (x, y, z)))
    s += "\n"
    return s


def distmat(a, index, size):
    # For each item in a
    # Find the difference on each axis and square it

    distances = np.zeros((size, 1))
    for i, point in enumerate(a):
        x, y, z = point

        distances[i] = np.sqrt((x - index[0]) ** 2 + (y - index[1]) ** 2 + (z - index[2]) ** 2)

    return distances


def convert_3d_2d(geometry):
    """
    Takes a GeoSeries of 3D Multi/Polygons (has_z) and returns a list of 2D Multi/Polygons

    :param geometry:
    :return:
    """
    new_geo = []
    for p in geometry:
        if p.has_z:
            if p.geom_type == 'Polygon':
                lines = [xy[:2] for xy in list(p.exterior.coords)]
                new_p = Polygon(lines)
                new_geo.append(new_p)
            elif p.geom_type == 'MultiPolygon':
                new_multi_p = []
                for ap in p:
                    lines = [xy[:2] for xy in list(ap.exterior.coords)]
                    new_p = Polygon(lines)
                    new_multi_p.append(new_p)
                new_geo.append(MultiPolygon(new_multi_p))
    return new_geo


def create_exclude_polygon(corner_points):
    ply = convert_3d_2d([Polygon(corner_points)])[0]
    triangles = triangulate(ply)

    verts = []
    for triangle in triangles:
        verts.extend(triangle.exterior.coords)

    faces = [(0, 1, 2), (3, 4, 5)]

    return trimesh.creation.extrude_triangulation(verts, faces, 300)


def get_cam_corners(camera):
    res = camera.resolution
    half_fov = np.radians(camera.fov) / 2.0

    right_top = np.tan(half_fov)
    # move half a pixel width in
    right_top *= 1 - (1.0 / res)
    left_bottom = -right_top
    # we are looking down the negative z axis, so
    # right_top corresponds to maximum x/y values
    # bottom_left corresponds to minimum x/y values

    right, top = right_top
    left, bottom = left_bottom
    xy = np.array([[left, top], [right, top], [left, bottom], [right, bottom]])

    return util.unitize(
        np.column_stack((xy, -np.ones_like(xy[:, :1]))))


class Viewer(SceneViewer):

    def on_mouse_press(self, x, y, buttons, modifiers):
        super(Viewer, self).on_mouse_press(x, y, buttons, modifiers)

