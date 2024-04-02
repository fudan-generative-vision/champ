import numpy as np
import torch
from torch.nn import functional as F
# from psbody.mesh.visibility import visibility_compute

def uv_to_xyz_and_normals(verts, f, fmap, bmap, ftov):
    vn = estimate_vertex_normals(verts, f, ftov)
    pixels_to_set = torch.nonzero(fmap+1)
    x_to_set = pixels_to_set[:,0]
    y_to_set = pixels_to_set[:,1]
    b_coords = bmap[x_to_set, y_to_set, :]
    f_coords = fmap[x_to_set, y_to_set]
    v_ids = f[f_coords]
    points = (b_coords[:,0,None]*verts[:,v_ids[:,0]]
             + b_coords[:,1,None]*verts[:,v_ids[:,1]]
             + b_coords[:,2,None]*verts[:,v_ids[:,2]])
    normals = (b_coords[:,0,None]*vn[:,v_ids[:,0]]
             + b_coords[:,1,None]*vn[:,v_ids[:,1]]
             + b_coords[:,2,None]*vn[:,v_ids[:,2]])
    return points, normals, vn, f_coords

def estimate_vertex_normals(v, f, ftov):
    face_normals = TriNormalsScaled(v, f)
    non_scaled_normals = torch.einsum('ij,bjk->bik', ftov, face_normals)
    norms = torch.sum(non_scaled_normals ** 2.0, 2) ** 0.5
    norms[norms == 0] = 1.0
    return torch.div(non_scaled_normals, norms[:,:,None])

def TriNormalsScaled(v, f):
    return torch.cross(_edges_for(v, f, 1, 0), _edges_for(v, f, 2, 0))

def _edges_for(v, f, cplus, cminus):
    return v[:,f[:,cplus]] - v[:,f[:,cminus]]

def psbody_get_face_visibility(v, n, f, cams, normal_threshold=0.5):
    bn, nverts, _ = v.shape
    nfaces, _ = f.shape
    vis_f = np.zeros([bn, nfaces], dtype='float32')
    for i in range(bn):
        vis, n_dot_cam = visibility_compute(v=v[i], n=n[i], f=f, cams=cams)
        vis_v = (vis == 1) & (n_dot_cam > normal_threshold)
        vis_f[i] = np.all(vis_v[0,f],1)
    return vis_f

def compute_uvsampler(vt, ft, tex_size=6):
    """
    For this mesh, pre-computes the UV coordinates for
    F x T x T points.
    Returns F x T x T x 2
    """
    uv = obj2nmr_uvmap(ft, vt, tex_size=tex_size)
    uv = uv.reshape(-1, tex_size, tex_size, 2)
    return uv

def obj2nmr_uvmap(ft, vt, tex_size=6):
    """
    Converts obj uv_map to NMR uv_map (F x T x T x 2),
    where tex_size (T) is the sample rate on each face.
    """
    # This is F x 3 x 2
    uv_map_for_verts = vt[ft]

    # obj's y coordinate is [1-0], but image is [0-1]
    uv_map_for_verts[:, :, 1] = 1 - uv_map_for_verts[:, :, 1]

    # range [0, 1] -> [-1, 1]
    uv_map_for_verts = (2 * uv_map_for_verts) - 1

    alpha = np.arange(tex_size, dtype=float) / (tex_size - 1)
    beta = np.arange(tex_size, dtype=float) / (tex_size - 1)
    import itertools
    # Barycentric coordinate values
    coords = np.stack([p for p in itertools.product(*[alpha, beta])])

    # Compute alpha, beta (this is the same order as NMR)
    v2 = uv_map_for_verts[:, 2]
    v0v2 = uv_map_for_verts[:, 0] - uv_map_for_verts[:, 2]
    v1v2 = uv_map_for_verts[:, 1] - uv_map_for_verts[:, 2]
    # Interpolate the vertex uv values: F x 2 x T*2
    uv_map = np.dstack([v0v2, v1v2]).dot(coords.T) + v2.reshape(-1, 2, 1)

    # F x T*2 x 2  -> F x T x T x 2
    uv_map = np.transpose(uv_map, (0, 2, 1)).reshape(-1, tex_size, tex_size, 2)

    return uv_map
