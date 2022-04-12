import sys

import numpy as np
from osgeo import ogr
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


class Viewer(SceneViewer):

    def on_mouse_press(self, x, y, buttons, modifiers):
        super(Viewer, self).on_mouse_press(x, y, buttons, modifiers)
        print(x, y, buttons, modifiers)
