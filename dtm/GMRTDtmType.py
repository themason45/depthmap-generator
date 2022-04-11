import uuid
from typing import Any

from dtm.dtm import DtmType, GeoTransform
from owslib.wms import WebMapService

from osgeo import gdal
import numpy as np


class GMRTDtmType(DtmType):
    """
    Global Multi-Resolution Topography (GMRT)
    https://www.gmrt.org/services/index.php

    Give this DTM Type its BBOX in metres
    The SRS format used here is EPSG:3857

    """
    wms_url = "https://www.gmrt.org/services/mapserver/wms_merc"
    srs = "EPSG:3857"
    fmt = "image/tiff"

    def get_wms(self) -> WebMapService:
        wms = WebMapService(self.wms_url, version='1.3.0')

        return wms

    def get_raster(self, bbox: (int, int, int, int)) -> gdal.Dataset | None:
        """

        :param bbox: Bounding box to get the map tile from, in the format (minx,miny,maxx,maxy)
        :return:
        """
        width = np.abs(bbox[2] - bbox[0])
        height = np.abs(bbox[3] - bbox[1])

        if width > self.resolution[0] or height > self.resolution[1]:
            print("Bounds too big")
            return None

        img = self.get_wms().getmap(
            layers=["topo"],
            styles='',
            srs=self.srs,
            format=self.fmt,
            bbox=bbox,
            transparent=False,
            size=[r * self.scale for r in self.resolution]
        )

        with open("hm.tiff", "wb") as f:
            f.write(img.read())

        # Load the TIFF into a memory map, that GDAL can then read
        mmap_name = f"/vsimem/{uuid.uuid4().hex}"
        gdal.FileFromMemBuffer(mmap_name, img.read())

        ds = gdal.Open(mmap_name)
        return ds
