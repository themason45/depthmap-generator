import uuid

import requests
from osgeo import gdal

from dtm.dtm import DtmType


class DefraDtmType(DtmType):
    srs = "3857"

    def __init__(self, bbox: (int, int, int, int), resolution: (int, int) = (2000, 2000), scale: float = 1):
        super().__init__(bbox, resolution, scale)

    def get_raster(self, bbox: (int, int, int, int)):
        # bbox fmt: (minx, miny, maxx, maxy)
        resp = requests.get(
            "https://environment.data.gov.uk/image/rest/services/SURVEY/LIDAR_Composite_1m_DTM_2020_Elevation/"
            "ImageServer/exportImage",
            params={
                "bbox": ",".join(str(x) for x in bbox),
                "bboxSR": self.srs,
                "imageSR": self.srs,
                "size": ",".join((str(x * self.scale) for x in self.resolution)),
                "format": "tiff",
                "transparent": "true",
                "f": "image"
            }
        )

        print(resp.url)

        with open("hm.tiff", "wb") as f:
            f.write(resp.content)

        import logging
        logging.basicConfig(level=logging.DEBUG)

        gdal.SetConfigOption("CPL_DEBUG", "ON")
        gdal.UseExceptions()
        gdal.ConfigurePythonLogging()

        # Load the TIFF into a memory map, that GDAL can then read
        mmap_name = f"/vsimem/{uuid.uuid4().hex}"
        gdal.FileFromMemBuffer(mmap_name, resp.content)

        ds = gdal.Open(mmap_name)
        return ds

