from abc import ABC
from dataclasses import dataclass

import PIL.Image
import numpy as np
import trimesh
from osgeo import gdal
from trimesh.visual import TextureVisuals


@dataclass(kw_only=False)
class GeoTransform:
    """
    Geo-Transform info can be found here:
    https://gdal.org/tutorials/geotransforms_tut.html
    """

    x_top_left: int
    x_pixel_size: float
    row_rot: float

    y_top_left: int
    col_rot: float
    y_pixel_size: float


class DtmType:
    """
    This class is a base for other Dtm classes.

    It basically serves as an interface that provides some methods, in particular, loading the mesh, and
    storing details about it, such as pixel size etc
    """
    resolution = (2000, 2000)
    scale = 1

    _raster: gdal.Dataset = None
    _trimesh: trimesh.Trimesh = None
    bbox: (int, int, int, int) = (0, 0, 0, 0)  # (minx,miny,maxx,maxy)

    def __init__(self, bbox: (int, int, int, int), resolution: (int, int) = (2000, 2000), scale: float = 1):
        self.bbox = bbox
        self.resolution = resolution
        self.scale = scale

        self._raster = self.get_raster(bbox)

    def get_raster(self, bbox: (int, int, int, int)):
        raise NotImplementedError("This method must be implemented")

    @property
    def raster(self):
        if self._raster is None:
            self._raster = self.get_raster(bbox=self.bbox)

        return self._raster

    @property
    def geo_transform(self):
        t: (int, float, float, int, float, float) = self.raster.GetGeoTransform()
        return GeoTransform(*t)

    def get_vertices(self) -> np.ndarray:
        transform = self.geo_transform
        width = self.raster.RasterXSize
        height = self.raster.RasterYSize

        x = np.arange(0, width) * transform.x_pixel_size + transform.x_top_left
        y = np.arange(0, height) * transform.y_pixel_size + transform.y_top_left
        xx, yy = np.meshgrid(x, y)

        zz = self.raster.GetRasterBand(1).ReadAsArray()

        vertices = np.vstack((xx, yy, zz))
        vertices = vertices.reshape([3, -1])
        vertices = vertices.transpose()

        return vertices

    def get_indices(self) -> np.ndarray:
        width = self.raster.RasterXSize
        height = self.raster.RasterYSize

        ai = np.arange(0, width - 1)
        aj = np.arange(0, height - 1)
        aii, ajj = np.meshgrid(ai, aj)
        a = aii + ajj * width
        a = a.flatten()

        tria = np.vstack((a, a + width, a + width + 1, a, a + width + 1, a + 1))
        tria = np.transpose(tria).reshape([-1, 3])
        return tria

    @property
    def trimesh(self):
        if self._trimesh is None:
            verts = self.get_vertices()
            faces = self.get_indices()

            return trimesh.Trimesh(vertices=verts.tolist(), faces=faces.tolist())

        return self._trimesh

    def get_visual(self):
        m = self.trimesh

        img = PIL.Image.open("heheh.png")
        uv = m.unwrap(image=None)

        return TextureVisuals(uv=uv, image=img)
