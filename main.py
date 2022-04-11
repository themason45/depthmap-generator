import csv
import faulthandler
import sys

import numpy as np
import trimesh
from pyproj import Proj, Transformer

from dtm.DefraDtmType import DefraDtmType
from dtm.camera import DTCamera
from dtm.helpers import generate_bbox, coord_string
from dtm.image import Image
from embreeintersector import RayMeshIntersector

import matplotlib as plt

P_EPSG4326 = Proj("epsg:4326")
P_EPSG3857 = Proj("epsg:3857")

if __name__ == '__main__':
    faulthandler.enable(file=sys.stderr, all_threads=False)

    with open("tmp/imageinfo.csv", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        keys, vals = reader
        img = Image({k: v for k, v in zip(keys, vals)})

        print(img.campos)

        transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
        coords = transformer.transform(*reversed(img.campos[:2]))

        print(coords)

        # coord = [-298097, 7008381]
        dtm = DefraDtmType(generate_bbox(*coords), scale=0.1)

        scene = trimesh.scene.scene.Scene()
        scene = scene.convert_units("m", guess=True)

        mesh = dtm.trimesh
        scene.add_geometry(mesh)

        cam = scene.camera
        cam.z_far = 100000

        scene.camera = cam
        intersector = RayMeshIntersector(mesh)

        m_cam = DTCamera(image=img, coords=coords)
        m_cam.resolution = [100, 100]

        h, _, _, _ = intersector.intersects_location([m_cam.cam_pt], [[0, 0, -1]])

        print(h[0, 2])
        m_cam.z_offset = h[0, 2] * 2
        vectors, _ = m_cam.to_rays()

        v = np.array(list(map(lambda v: img.rs_matrix()[:3, :3] @ v, vectors)))
        o = np.tile(m_cam.cam_pt, (v.shape[0], 1))

        locs, dists, idx_ray, idx_tri = intersector.intersects_location(o, v, multiple_hits=False)

        scene.add_geometry(trimesh.points.PointCloud(locs)) if not len(locs) == 0 else None

        list(scene.add_geometry(m) for m in m_cam.marker)

        print(dists)
        d = dists.reshape((100, 100))
        plt.imshow(d, interpolation='nearest')
        plt.show()

        faulthandler.disable()

        # with open("out.obj", "w") as f1:
        #     f1.write(export_obj(mesh))

        a = trimesh.creation.axis(axis_length=100, axis_radius=5)

        print(a)
        a.apply_translation(m_cam.cam_pt)
        scene.add_geometry(a)

        with open("out.xyz", "w") as f2:
            f2.write(coord_string(*m_cam.cam_pt))
            for l in locs:
                f2.write(coord_string(*l))

            print(f"Wrote {len(locs)} pts")

        # viewer = Viewer(scene)
