"""
Ray queries using the pyembree package with the
API wrapped to match our native raytracer.
"""

from copy import deepcopy

import embree
import numpy as np
from trimesh import caching
from trimesh import intersections
from trimesh import util
from trimesh.constants import log_time
from trimesh.ray.ray_util import contains_points

# the factor of geometry.scale to offset a ray from a triangle
# to reliably not hit its origin triangle
_ray_offset_factor = 1e-4
# we want to clip our offset to a sane distance
_ray_offset_floor = 1e-8

# see if we're using a newer version of the pyembree wrapper
_embree_new = True
# both old and new versions require exact but different type
_embree_dtype = [np.float64, np.float32][int(_embree_new)]


class RayMeshIntersector(object):

    def __init__(self,
                 geometry,
                 scale_to_box=True):
        """
        Do ray- mesh queries.

        Parameters
        -------------
        geometry : Trimesh object
          Mesh to do ray tests on
        scale_to_box : bool
          If true, will scale mesh to approximate
          unit cube to avoid problems with extreme
          large or small meshes.
        """
        self.mesh = geometry
        self._scale_to_box = scale_to_box
        self._cache = caching.Cache(id_function=self.mesh.crc)

    @property
    def _scale(self):
        """
        Scaling factor for precision.
        """
        if self._scale_to_box:
            # scale vertices to approximately a cube to help with
            # numerical issues at very large/small scales
            scale = 100.0 / self.mesh.scale
        else:
            scale = 1.0
        return scale

    @caching.cache_decorator
    def _scene(self):
        """
        A cached version of the pyembree scene.
        """
        return _EmbreeWrap(vertices=self.mesh.vertices,
                           faces=self.mesh.faces,
                           scale=self._scale)

    def intersects_location(self,
                            ray_origins,
                            ray_directions,
                            multiple_hits=True):
        """
        Return the location of where a ray hits a surface.

        Parameters
        ----------
        ray_origins : (n, 3) float
          Origins of rays
        ray_directions : (n, 3) float
          Direction (vector) of rays

        Returns
        ---------
        locations : (m) sequence of (p, 3) float
          Intersection points
        index_ray : (m,) int
          Indexes of ray
        index_tri : (m,) int
          Indexes of mesh.faces
          :param ray_directions:
          :param ray_origins:
          :param multiple_hits:
        """
        (index_tri,
         index_ray,
         locations, distances) = self.intersects_id(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=multiple_hits,
            return_locations=True)

        return locations, distances, index_ray, index_tri

    @log_time
    def intersects_id(self,
                      ray_origins,
                      ray_directions,
                      multiple_hits=True,
                      max_hits=20,
                      return_locations=False):
        """
        Find the triangles hit by a list of rays, including
        optionally multiple hits along a single ray.


        Parameters
        ----------
        ray_origins : (n, 3) float
          Origins of rays
        ray_directions : (n, 3) float
          Direction (vector) of rays
        multiple_hits : bool
          If True will return every hit along the ray
          If False will only return first hit
        max_hits : int
          Maximum number of hits per ray
        return_locations : bool
          Should we return hit locations or not

        Returns
        ---------
        index_tri : (m,) int
          Indexes of mesh.faces
        index_ray : (m,) int
          Indexes of ray
        locations : (m) sequence of (p, 3) float
          Intersection points, only returned if return_locations
        """
        # make sure input is _dtype for embree
        ray_origins = np.asanyarray(
            deepcopy(ray_origins),
            dtype=np.float64)
        ray_directions = np.asanyarray(ray_directions,
                                       dtype=np.float64)
        ray_directions = util.unitize(ray_directions)

        # since we are constructing all hits, save them to a deque then
        # stack into (depth, len(rays)) at the end
        result_triangle = []
        result_ray_idx = []
        result_locations = []
        result_distances = []

        # the mask for which rays are still active
        current = np.ones(len(ray_origins), dtype=bool)

        if multiple_hits or return_locations:
            # how much to offset ray to transport to the other side of face
            distance = np.clip(_ray_offset_factor * self._scale,
                               _ray_offset_floor,
                               np.inf)
            ray_offsets = ray_directions * distance

            # grab the planes from triangles
            plane_origins = self.mesh.triangles[:, 0, :]
            plane_normals = self.mesh.face_normals

        # use a for loop rather than a while to ensure this exits
        # if a ray is offset from a triangle and then is reported
        # hitting itself this could get stuck on that one triangle
        for query_depth in range(max_hits):
            # run the pyembree query
            # if you set output=1 it will calculate distance along
            # ray, which is bizzarely slower than our calculation

            query, distances = self._scene.run(
                ray_origins[current],
                ray_directions[current])  # type: np.ndarray

            # basically we need to reduce the rays to the ones that hit
            # something
            hit = query < len(self.mesh.faces)
            # which triangle indexes were hit
            hit_triangle = query[hit]
            # eliminate rays that didn't hit anything from future queries
            current_index = np.nonzero(current)[0]
            current_index_no_hit = current_index[np.logical_not(hit)]
            current_index_hit = current_index[hit]
            current[current_index_no_hit] = False

            # append the triangle and ray index to the results
            result_triangle.append(hit_triangle)
            result_ray_idx.append(current_index_hit)
            result_distances.append(distances)

            # if we don't need all of the hits, return the first one
            if ((not multiple_hits and
                 not return_locations) or
                    not hit.any()):
                break

            # find the location of where the ray hit the triangle plane
            new_origins, valid = intersections.planes_lines(
                plane_origins=plane_origins[hit_triangle],
                plane_normals=plane_normals[hit_triangle],
                line_origins=ray_origins[current],
                line_directions=ray_directions[current])

            if not valid.all():
                # since a plane intersection was invalid we have to go back and
                # fix some stuff, we pop the ray index and triangle index,
                # apply the valid mask then append it right back to keep our
                # indexes intact
                result_ray_idx.append(result_ray_idx.pop()[valid])
                result_triangle.append(result_triangle.pop()[valid])

                # update the current rays to reflect that we couldn't find a
                # new origin
                current[current_index_hit[np.logical_not(valid)]] = False

            # since we had to find the intersection point anyway we save it
            # even if we're not going to return it
            result_locations.extend(new_origins)

            if multiple_hits:
                # move the ray origin to the other side of the triangle
                ray_origins[current] = new_origins + ray_offsets[current]
            else:
                break

        # stack the deques into nice 1D numpy arrays
        index_tri = np.hstack(result_triangle)
        index_ray = np.hstack(result_ray_idx)
        distances = np.hstack(result_distances)

        if return_locations:
            locations = (
                np.zeros((0, 3), float) if len(result_locations) == 0
                else np.array(result_locations))

            return index_tri, index_ray, locations, distances
        return index_tri, index_ray, distances

    @log_time
    def intersects_first(self,
                         ray_origins,
                         ray_directions):
        """
        Find the index of the first triangle a ray hits.

        Parameters
        ----------
        ray_origins : (n, 3) float
          Origins of rays
        ray_directions : (n, 3) float
          Direction (vector) of rays

        Returns
        ----------
        triangle_index : (n,) int
          Index of triangle ray hit, or -1 if not hit
        """

        ray_origins = np.asanyarray(deepcopy(ray_origins))
        ray_directions = np.asanyarray(ray_directions)

        triangle_index = self._scene.run(ray_origins,
                                         ray_directions)
        return triangle_index

    def intersects_any(self,
                       ray_origins,
                       ray_directions):
        """
        Check if a list of rays hits the surface.


        Parameters
        -----------
        ray_origins : (n, 3) float
          Origins of rays
        ray_directions : (n, 3) float
          Direction (vector) of rays

        Returns
        ----------
        hit : (n,) bool
          Did each ray hit the surface
        """

        first = self.intersects_first(ray_origins=ray_origins,
                                      ray_directions=ray_directions)
        hit = first != -1
        return hit

    def contains_points(self, points):
        """
        Check if a mesh contains a list of points, using ray tests.

        If the point is on the surface of the mesh, behavior is undefined.

        Parameters
        ---------
        points: (n, 3) points in space

        Returns
        ---------
        contains: (n,) bool
                         Whether point is inside mesh or not
        """
        return contains_points(self, points)


class _EmbreeWrap(object):
    """
    A light wrapper for PyEmbree scene objects which
    allows queries to be scaled to help with precision
    issues, as well as selecting the correct dtypes.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, scale):
        self.verts = vertices
        self.faces = faces
        scaled = np.array(vertices,
                          dtype=np.float64)
        # self.origin = np.amin(scaled)
        # self.scale = float(scale)
        # scaled = (scaled - self.origin) * self.scale

        self.device = embree.Device()

        self.scene = self.device.make_scene()  # type: embree.Scene
        # assign the geometry to the scene
        self.mesh = embree.Geometry(self.device, embree.GeometryType.Triangle)

        # Set vertices
        _verts = self.mesh.set_new_buffer(embree.BufferType.Vertex, 0, embree.Format.Float3,
                                          3 * np.dtype('float32').itemsize,
                                          vertices.shape[0])
        _verts[0:] = vertices

        # Set indices
        _faces = self.mesh.set_new_buffer(embree.BufferType.Index, 0, embree.Format.Uint3,
                                          3 * np.dtype('uint32').itemsize, faces.shape[0])
        _faces[0:] = faces

        # Commit array
        self.mesh.commit()
        self.scene.attach_geometry(self.mesh)
        self.mesh.release()

        self.scene.commit()

    def run(self, origins, normals):
        # scaled = (np.array(origins,
        #                    dtype=np.float64) - self.origin) * self.scale
        ray_count = origins.shape[0]

        rh = embree.RayHit1M(ray_count)

        rh.tnear[:] = 0
        rh.tfar[:] = np.inf
        rh.time[:] = 0
        rh.prim_id[:] = embree.INVALID_GEOMETRY_ID
        rh.geom_id[:] = embree.INVALID_GEOMETRY_ID

        np.copyto(rh.org, origins)
        np.copyto(rh.dir, normals)

        context = embree.IntersectContext()
        context.flags = embree.IntersectContextFlags.COHERENT

        self.scene.intersect1M(context, rh)

        print(rh.tfar)

        return rh.prim_id, rh.tfar

    def close(self):
        self.scene.release()
        self.device.release()

    def __del__(self):
        self.close()
