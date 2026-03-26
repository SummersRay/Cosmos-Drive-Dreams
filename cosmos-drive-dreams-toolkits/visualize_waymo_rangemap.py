"""
Visualize Waymo range maps converted for LiDAR tokenizer training.
Loads sparse tar files and renders range map videos with colormap.
Supports point cloud rendering via plotly.
"""

import argparse
import io
import json
import os
import tarfile
from multiprocessing import Pool

import numpy as np
import torch
import mediapy as media
from matplotlib import cm
import plotly.graph_objects as go
from PIL import Image
from tqdm import tqdm


def load_each_frame_from_tar_data(tar_data, frame_idx, n_rows=128, n_cols=3600):
    lidar_row = np.load(tar_data.extractfile(f'{frame_idx}.lidar_row.npz'))['arr_0']
    lidar_col = np.load(tar_data.extractfile(f'{frame_idx}.lidar_col.npz'))['arr_0']
    lidar_range = np.load(tar_data.extractfile(f'{frame_idx}.lidar_range.npz'))['arr_0']
    range_map = np.zeros((n_rows, n_cols), dtype=np.float32)
    range_map[lidar_row, lidar_col] = lidar_range.astype(np.float32)
    return range_map


def load_range_map(tar_file, n_rows=128, n_cols=3600):
    frame_idx_list = sorted(
        [x.strip(".lidar_row.npz") for x in tar_file.getnames() if "lidar_row" in x]
    )
    range_map_list = []
    for frame_idx in frame_idx_list:
        range_map = load_each_frame_from_tar_data(tar_file, frame_idx, n_rows, n_cols)
        range_map_list.append(range_map)
    return np.stack(range_map_list, axis=0)


def colorcode_depth_maps(result, near=None, far=None, cmap="Spectral"):
    """
    Input: B x H x W (torch.Tensor)
    Output: B x 3 x H x W, normalized to [0, 1]
    """
    mask = result == 0
    n_frames = result.shape[0]
    if far is None:
        far = result[n_frames // 2].view(-1).quantile(0.99).log()
    if near is None:
        valid = result[n_frames // 2][result[n_frames // 2] > 0]
        if valid.numel() > 0:
            near = valid.quantile(0.01).log()
        else:
            near = torch.zeros_like(far)

    result = result.clone()
    result[mask] = 1.0  # avoid log(0)
    result = result.log()
    result = 1 - (result - near) / (far - near)
    result = result.clip(0, 1)

    # apply colormap
    cmap_fn = cm.get_cmap(cmap)
    mapped = cmap_fn(result.cpu().numpy())[..., :3]  # B x H x W x 3
    image = torch.tensor(mapped, dtype=result.dtype)
    # set masked areas to light gray
    image[mask] = torch.tensor([0.82, 0.82, 0.82], dtype=image.dtype)
    # B x H x W x 3 -> B x 3 x H x W
    image = image.permute(0, 3, 1, 2)
    return image


def save_depth_maps_to_video(depth_maps, save_path, cmap="Spectral", fps=10):
    """
    depth_maps: B x H x W (torch.Tensor)
    """
    colored = colorcode_depth_maps(depth_maps, cmap=cmap)  # B x 3 x H x W
    colored = (colored * 255 + 0.5).permute(0, 2, 3, 1).numpy().astype(np.uint8)  # B x H x W x 3
    media.write_video(save_path, colored, fps=fps)
    print(f"Saved video: {save_path}")


def save_single_frame_image(depth_map, save_path, cmap="Spectral"):
    """
    depth_map: H x W (numpy)
    """
    depth_tensor = torch.from_numpy(depth_map).float().unsqueeze(0)  # 1 x H x W
    colored = colorcode_depth_maps(depth_tensor, cmap=cmap)  # 1 x 3 x H x W
    image = (colored[0] * 255 + 0.5).permute(1, 2, 0).numpy().astype(np.uint8)  # H x W x 3
    media.write_image(save_path, image)
    print(f"Saved image: {save_path}")


CAMERA_VIEWS = {
    "front_view": {
        "eye": {"x": -0.3, "y": 0, "z": 0.2},
        "center": {"x": 0.1, "y": 0, "z": 0},
    },
    "top_down_view": {
        "eye": {"x": 0, "y": -0.05, "z": 0.5},
        "center": {"x": 0, "y": -0.05, "z": 0},
    },
}


def load_sensor_elevation_angles(param_path="assets/row-offset-spinning-lidar-model-parameters.json"):
    param = json.load(open(param_path, "r"))
    return np.rad2deg(np.array(param['row_elevations_rad']))


def range_map_to_ray_directions(n_cols, sensor_elevation_angles):
    azimuth_angles = np.linspace(np.pi, -np.pi, n_cols, endpoint=False)
    elevation_angles_rad = np.radians(sensor_elevation_angles)
    elevation_grid, azimuth_grid = np.meshgrid(elevation_angles_rad, azimuth_angles, indexing="ij")
    x = np.cos(elevation_grid) * np.cos(azimuth_grid)
    y = np.cos(elevation_grid) * np.sin(azimuth_grid)
    z = np.sin(elevation_grid)
    return np.stack([x, y, z], axis=-1)  # (H, W, 3)


def render_point_cloud_plotly(points, colors_rgb, camera_view, point_size=0.3,
                               width=1280, height=720, view_range=100):
    """Render point cloud to image using plotly."""
    # Convert RGB floats to plotly color strings
    colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in colors_rgb]

    trace = go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=point_size, color=colors, opacity=1.0, line=dict(width=0)),
    )
    fig = go.Figure(data=[trace])
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-view_range, view_range], autorange=False,
                       showbackground=False, showticklabels=False, zeroline=False, visible=False, showgrid=False),
            yaxis=dict(range=[-view_range, view_range], autorange=False,
                       showbackground=False, showticklabels=False, zeroline=False, visible=False, showgrid=False),
            zaxis=dict(range=[-view_range, view_range], autorange=False,
                       showbackground=False, showticklabels=False, zeroline=False, visible=False, showgrid=False),
            aspectmode="cube",
            camera=camera_view,
        ),
        paper_bgcolor="rgba(0,0,0)",
        plot_bgcolor="rgba(0,0,0)",
        margin=dict(l=0, r=0, b=0, t=0),
    )
    fig_bytes = fig.to_image(width=width, height=height)
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)[:, :, :3]


def render_single_frame(args):
    """Worker function for parallel rendering."""
    frame_idx, range_map_frame, ray_directions, camera_view, save_path = args
    valid_mask = range_map_frame > 0
    points = ray_directions[valid_mask] * range_map_frame[valid_mask][:, np.newaxis]

    # Color by height (z)
    z = points[:, 2]
    z_clipped = np.clip(z, -2, 4)
    z_norm = (z_clipped + 2) / 6.0
    cmap_fn = cm.get_cmap("rainbow")
    colors_rgb = cmap_fn(z_norm)[:, :3]

    img = render_point_cloud_plotly(points, colors_rgb, camera_view)
    Image.fromarray(img).save(save_path)
    return save_path


def render_point_cloud_video(range_maps, output_path, sensor_elevation_angles,
                              camera_view_name="front_view", max_workers=8, fps=10):
    """Render range maps as point cloud video."""
    n_frames, h, w = range_maps.shape
    camera_view = CAMERA_VIEWS[camera_view_name]
    ray_directions = range_map_to_ray_directions(w, sensor_elevation_angles)

    tmp_dir = output_path.replace(".mp4", "_pcd_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    process_args = [
        (i, range_maps[i], ray_directions, camera_view,
         os.path.join(tmp_dir, f"pcd_{i:04d}.png"))
        for i in range(n_frames)
    ]

    with Pool(processes=max_workers) as pool:
        list(tqdm(pool.imap(render_single_frame, process_args),
                  total=n_frames, desc="Rendering point clouds"))

    # Combine frames into video
    frames = []
    for i in range(n_frames):
        img = np.array(Image.open(os.path.join(tmp_dir, f"pcd_{i:04d}.png")))
        frames.append(img)
    media.write_video(output_path, frames, fps=fps)
    print(f"Saved point cloud video: {output_path}")

    # Cleanup frame images
    for i in range(n_frames):
        os.remove(os.path.join(tmp_dir, f"pcd_{i:04d}.png"))
    os.rmdir(tmp_dir)


def main():
    parser = argparse.ArgumentParser(description="Visualize Waymo range maps")
    parser.add_argument("--tar_path", type=str, required=True, help="Path to lidar tar file")
    parser.add_argument("--output_dir", type=str, default="/data2/waymo_rangemap_vis", help="Output directory")
    parser.add_argument("--max_frames", type=int, default=-1, help="Max frames to visualize (-1 for all)")
    parser.add_argument("--fps", type=int, default=10, help="FPS for video output")
    parser.add_argument("--colormap", type=str, default="Spectral", help="Colormap name")
    parser.add_argument("--save_frames", action="store_true", help="Also save individual frame images")
    parser.add_argument("--vis_pcd", action="store_true", help="Render point cloud video")
    parser.add_argument("--camera_view", type=str, default="front_view",
                        choices=list(CAMERA_VIEWS.keys()), help="Camera view for point cloud rendering")
    parser.add_argument("--max_workers", type=int, default=8, help="Max parallel workers for PCD rendering")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    scene_name = os.path.basename(args.tar_path).replace(".tar", "")
    print(f"Loading range maps from: {args.tar_path}")

    tar_handle = tarfile.open(args.tar_path, "r")
    range_maps = load_range_map(tar_handle)  # N x 128 x 3600
    tar_handle.close()

    if args.max_frames > 0:
        range_maps = range_maps[:args.max_frames]

    n_frames, h, w = range_maps.shape
    valid_pixels = (range_maps > 0).sum()
    total_pixels = n_frames * h * w
    print(f"Scene: {scene_name}")
    print(f"Frames: {n_frames}, Shape: {h}x{w}")
    print(f"Valid pixels: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
    print(f"Range stats (valid only): min={range_maps[range_maps>0].min():.2f}m, "
          f"max={range_maps[range_maps>0].max():.2f}m, "
          f"mean={range_maps[range_maps>0].mean():.2f}m")

    # Save video
    range_tensor = torch.from_numpy(range_maps).float()
    video_path = os.path.join(args.output_dir, f"{scene_name}.mp4")
    save_depth_maps_to_video(range_tensor, video_path, cmap=args.colormap, fps=args.fps)

    # Optionally save individual frames
    if args.save_frames:
        frame_dir = os.path.join(args.output_dir, scene_name)
        os.makedirs(frame_dir, exist_ok=True)
        for i in range(n_frames):
            frame_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
            save_single_frame_image(range_maps[i], frame_path, cmap=args.colormap)

    # Render point cloud video
    if args.vis_pcd:
        param_path = os.path.join(os.path.dirname(__file__), "assets",
                                  "row-offset-spinning-lidar-model-parameters.json")
        sensor_elevation_angles = load_sensor_elevation_angles(param_path)
        pcd_video_path = os.path.join(args.output_dir, f"{scene_name}_pcd.mp4")
        render_point_cloud_video(range_maps, pcd_video_path, sensor_elevation_angles,
                                  camera_view_name=args.camera_view,
                                  max_workers=args.max_workers, fps=args.fps)


if __name__ == "__main__":
    main()
