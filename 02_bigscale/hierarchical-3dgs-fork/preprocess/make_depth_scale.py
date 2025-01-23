import numpy as np
import argparse
import cv2
from joblib import delayed, Parallel
import json
from read_write_model import *

def get_scales(key, cameras, images, points3d_ordered, args):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    pts_idx = images_metas[key].point3D_ids

    mask = pts_idx >= 0
    mask *= pts_idx < len(points3d_ordered)

    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    if len(pts_idx) > 0:
        pts = points3d_ordered[pts_idx]
    else:
        pts = np.array([0, 0, 0])

    R = qvec2rotmat(image_meta.qvec) #读取出R T
    pts = np.dot(pts, R.T) + image_meta.tvec

    invcolmapdepth = 1. / pts[..., 2] # sfm的逆深度D_sfm,类似是float64
    n_remove = len(image_meta.name.split('.')[-1]) + 1 # 计算需要去除后缀字符串的数量
    invmonodepthmap = cv2.imread(f"{args.depths_dir}/{image_meta.name[:-n_remove]}.png", cv2.IMREAD_UNCHANGED) # D.
    
    if invmonodepthmap is None:
        return None
    
    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0] # 只取一个维度，灰度图三个维度都是一样的

    invmonodepthmap = invmonodepthmap.astype(np.float32) / (2**16)  # 这里有一些问题，原来的深度图是用uint8来保存，这里除了uint16的大小，归一化是没有意义，这也不是逆深度，做的是分子啊
    s = invmonodepthmap.shape[0] / cam_intrinsic.height

    maps = (valid_xys * s).astype(np.float32)
    valid = (  # 检查有效性
        (maps[..., 0] >= 0) * 
        (maps[..., 1] >= 0) * 
        (maps[..., 0] < cam_intrinsic.width * s) * 
        (maps[..., 1] < cam_intrinsic.height * s) * (invcolmapdepth > 0))
    
    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        invcolmapdepth = invcolmapdepth[valid] 
        invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0] # D
        
        ## Median / dev
        t_colmap = np.median(invcolmapdepth)                  #t(Dsfm)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap)) #s (Dsfm)

        t_mono = np.median(invmonodepth)   # t(D)
        s_mono = np.mean(np.abs(invmonodepth - t_mono)) #s(D)
        scale = s_colmap / s_mono     #s(D_sfm)/s(D)
        offset = t_colmap - t_mono * scale  #t(Dsfm) - s(D_sfm)/s(D)
    else:
        scale = 0
        offset = 0
    return {"image_name": image_meta.name[:-n_remove], "scale": scale, "offset": offset}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--depths_dir', required=True)
    parser.add_argument('--model_type', default="bin")
    args = parser.parse_args()

    # 读取每个chunk中sparse中的几个文件参数，
    # colmap生成的num_cameras: 1
    # num_images: 1500
    # num_points3D: 184809,字典类型
    cam_intrinsics, images_metas, points3d = read_model(os.path.join(args.base_dir, "sparse", "0"), ext=f".{args.model_type}")

    pts_indices = np.array([points3d[key].id for key in points3d])  # 得到key，也就是point中的id
    pts_xyzs = np.array([points3d[key].xyz for key in points3d])
    points3d_ordered = np.zeros([pts_indices.max()+1, 3])
    points3d_ordered[pts_indices] = pts_xyzs  # pts_indices 作为索引，将 pts_xyzs 中的坐标填充到 points3d_ordered 矩阵的对应行中。


    # depth_param_list = [get_scales(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas]
    depth_param_list = Parallel(n_jobs=-1, backend="threading")(
        delayed(get_scales)(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas
    )

    depth_params = {
        depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
        for depth_param in depth_param_list if depth_param != None
    }

    with open(f"{args.base_dir}/sparse/0/depth_params.json", "w") as f:
        json.dump(depth_params, f, indent=2)

    print(0)
