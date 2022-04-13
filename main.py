#!./venv/bin/python

import csv
import faulthandler

from osgeo import gdal

from dtm.DefraDtmType import DefraDtmType

import numpy as np
import trimesh
from pyproj import Proj, Transformer

from dtm.camera import DTCamera
from dtm.helpers import generate_bbox, coord_string, distmat
from dtm.image import Image
from raytrace.embreeintersector import RayMeshIntersector

from matplotlib import pyplot as plt
import PIL

P_EPSG4326 = Proj("epsg:4326")
P_EPSG3857 = Proj("epsg:3857")

if __name__ == '__main__':
    faulthandler.enable()
    gdal.UseExceptions()

    with open("tmp/imageinfo.csv", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        keys, vals = reader
        img = Image({k: v for k, v in zip(keys, vals)})

        transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
        coords = transformer.transform(*reversed(img.campos[:2]))

        # coord = [-298097, 7008381]
        dtm = DefraDtmType(generate_bbox(*coords), scale=0.1)

        scene = trimesh.scene.scene.Scene()
        scene = scene.convert_units("m", guess=True)

        mesh = dtm.trimesh
        scene.add_geometry(mesh)

        cam = scene.camera
        cam.z_far = 100000

        scene.camera = cam
        pre_intersector = RayMeshIntersector(mesh)

        m_cam = DTCamera(image=img, coords=coords)
        m_cam.resolution = [img.width * 0.15, img.height * 0.15]

        h, _, _, _ = pre_intersector.intersects_location([m_cam.cam_pt], [[0, 0, -1]])

        # TODO: For efficiency improvements, we can cull most of the mesh that we don't need to perform queries on

        # pre_o = np.tile(m_cam.cam_pt, (4, 1))  # Create for origin points
        # pre_v = get_cam_corners(m_cam)
        #
        # print(pre_v)
        #
        # locs, _, _, _ = pre_intersector.intersects_location(pre_o, pre_v, multiple_hits=False)
        # # print(faces)
        #
        # ep = create_exclude_polygon(locs)
        #
        # scene.add_geometry(ep)
        #
        # submesh = mesh.slice_plane(ep.facets_origin, ep.facets_normal)
        # # submesh.show()

        intersector = RayMeshIntersector(mesh)

        m_cam.z_offset = h[0, 2] * 2
        vectors, pixels = m_cam.to_rays()

        v = np.array(list(map(lambda v: img.rs_matrix()[:3, :3] @ v, vectors)))
        o = np.tile(m_cam.cam_pt, (v.shape[0], 1))

        locs, dists, idx_ray, idx_tri = intersector.intersects_location(o, v, multiple_hits=False)

        scene.add_geometry(trimesh.points.PointCloud(locs)) if not len(locs) == 0 else None

        list(scene.add_geometry(m) for m in m_cam.marker)

        fig, axs = plt.subplots(ncols=3, figsize=(15, 5))

        print("Calculating euclid")
        d = distmat(locs, m_cam.cam_pt, np.prod(m_cam.resolution))
        d = d.reshape(np.flip(m_cam.resolution))
        d = np.flip(d, axis=1)
        axs[0].set_title("Euclidian distance")
        axs[0].imshow(d, origin="lower")

        print("Manipulating tFar")
        d = dists.reshape(np.flip(m_cam.resolution))
        d = np.flip(d, axis=1)
        axs[1].set_title("tFar values from embree")
        axs[1].imshow(d, origin="lower")

        print("Calculating depths")
        depth = trimesh.util.diagonal_dot(locs - o[0],
                                          v[idx_ray])

        pixel_ray = pixels[idx_ray]

        # create a numpy array we can turn into an image
        # doing it with uint8 creates an `L` mode greyscale image
        a = np.zeros(np.flip(m_cam.resolution), dtype=np.uint16)

        # scale depth against range (0.0 - 1.0)
        depth_float = ((depth - depth.min()) / depth.ptp())

        # convert depth into 0 - 255 uint8
        uint16_max = np.iinfo(np.uint16).max

        depth_int = (depth_float * uint16_max).round().astype(np.uint16)
        # assign depth to correct pixel locations
        # a[pixel_ray[:, 1], pixel_ray[:, 0]] = depth_int
        a[pixel_ray[:, 1], pixel_ray[:, 0]] = np.round(depth)
        print(depth.dtype)

        axs[2].set_title("depth from internet example")
        axs[2].imshow(a, origin="lower", interpolation="nearest", vmin=0, vmax=depth.max())

        # plt.show()

        img = PIL.Image.fromarray(np.flip(a, axis=0).astype(np.float32), mode="F")

        # show the resulting image
        img.save("depthmap.tiff", "TIFF")

        faulthandler.disable()

        # with open("out.obj", "w") as f1:
        #     f1.write(export_obj(mesh))

        a = trimesh.creation.axis(axis_length=100, axis_radius=5)

        a.apply_translation(m_cam.cam_pt)
        scene.add_geometry(a)

        with open("out.xyz", "w") as f2:
            f2.write(coord_string(*m_cam.cam_pt))
            for l in locs:
                f2.write(coord_string(*l))

            print(f"Wrote {len(locs)} pts")

        # viewer = Viewer(scene)
