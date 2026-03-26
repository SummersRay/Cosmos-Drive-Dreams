"""
Convert Waymo rds_hq lidar_raw data to LiDAR tokenizer training format.

Waymo rds_hq lidar_raw contains 'xyz' in vehicle frame and 'lidar_to_world'
(actually vehicle_to_world). This script:
  1. Transforms xyz from vehicle frame to LiDAR sensor frame (needed because
     the 2.184m height offset changes elevation angles significantly)
  2. Projects to 128x3600 range maps using real beam inclinations
  3. Composes pose as lidar_sensor_to_world = vehicle_to_world @ lidar_extrinsic

Output format:
  metadata/{clip_id}.npz  - pose_list (lidar_to_world pairs), timestamps_list, frame_indices
  lidar/{clip_id}.tar     - sparse range maps (row/col/range per frame)

Usage:
  python convert_waymo_lidar_to_tokenizer_format.py \
      --input_root /data2/rds_hq_waymo/training \
      --output_root /data2/rds_hq_waymo/lidar_tokenizer/training
"""

import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import click
import numpy as np
from pathlib import Path
from tqdm import tqdm
from webdataset import WebDataset, non_empty, TarWriter


# Waymo TOP LiDAR calibration (consistent across all Waymo Open Dataset clips)
# Extracted from frame.context.laser_calibrations[TOP].beam_inclinations
# 64 beams, non-uniform, in radians, ordered from highest to lowest elevation
WAYMO_TOP_BEAM_INCLINATIONS_RAD = np.array([
     0.03849,  0.03570,  0.03231,  0.02941,  0.02658,  0.02379,  0.02089,
     0.01803,  0.01460,  0.01198,  0.00899,  0.00617,  0.00329,  0.00055,
    -0.00268, -0.00545, -0.00849, -0.01113, -0.01419, -0.01682, -0.02016,
    -0.02294, -0.02590, -0.02894, -0.03222, -0.03554, -0.03962, -0.04336,
    -0.04745, -0.05171, -0.05606, -0.06060, -0.06619, -0.07076, -0.07635,
    -0.08161, -0.08721, -0.09276, -0.09927, -0.10564, -0.11206, -0.11833,
    -0.12536, -0.13210, -0.13941, -0.14685, -0.15429, -0.16180, -0.16976,
    -0.17758, -0.18603, -0.19431, -0.20291, -0.21142, -0.22096, -0.22990,
    -0.23938, -0.24864, -0.25816, -0.26761, -0.27795, -0.28828, -0.29886,
    -0.30935,
], dtype=np.float64)

# Waymo TOP LiDAR extrinsic: lidar_sensor_frame -> vehicle_frame
# LiDAR is mounted 1.43m forward, 0m lateral, 2.184m above vehicle origin
# Rotation is ~148° yaw + negligible tilt (~0.13°)
WAYMO_TOP_LIDAR_EXTRINSIC = np.array([
    [-8.4777248e-01, -5.3035414e-01, -2.5136571e-03,  1.4299999e+00],
    [ 5.3035545e-01, -8.4777534e-01,  1.8014426e-04,  0.0000000e+00],
    [-2.2265569e-03, -1.1804104e-03,  9.9999684e-01,  2.1840000e+00],
    [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00],
], dtype=np.float64)


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def get_sample(url):
    dataset = WebDataset(str(url), nodesplitter=non_empty, workersplitter=None, shardshuffle=False).decode()
    return next(iter(dataset))


def write_to_tar(sample, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sink = TarWriter(str(output_file))
    sink.write(sample)
    sink.close()


def make_elevation_angles(beam_inclinations_rad, n_rows):
    """Create n_rows elevation angles by interpolating Waymo TOP beams.

    Returns descending order (high-to-low), matching Pandar128 convention:
    row 0 = highest elevation (+2.21°), row N-1 = lowest (-17.72°).
    """
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")

    beam_deg = np.rad2deg(beam_inclinations_rad)
    beam_asc = beam_deg[::-1]  # ascending for interpolation
    x_orig = np.linspace(0, 1, len(beam_asc))
    x_new = np.linspace(0, 1, n_rows)
    elevation = np.interp(x_new, x_orig, beam_asc)
    return elevation[::-1].copy()  # descending: +2.21 ... -17.72


def points_to_range_map(xyz, n_rows, n_cols, elevation_angles_deg, max_range=105):
    """
    Project xyz points to a range map using spherical coordinates.

    Args:
        xyz: (N, 3) float32 array, points in LiDAR sensor frame
        n_rows: number of rows (elevation bins)
        n_cols: number of columns (azimuth bins)
        elevation_angles_deg: (n_rows,) array of elevation angles in degrees
        max_range: maximum range threshold

    Returns:
        range_map: (n_rows, n_cols) float32
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    range_values = np.sqrt(x**2 + y**2 + z**2)

    # Filter by range
    valid_range = (range_values > 0) & (range_values < max_range)

    # Compute azimuth -> column index (modulo wrap at 360°)
    azimuth = -np.arctan2(y, x) + np.pi  # [0, 2*pi)
    col_idx = ((azimuth / (2 * np.pi)) * n_cols).astype(np.int32) % n_cols

    # Compute elevation -> row index (find nearest elevation angle)
    elevation = np.arcsin(z / np.clip(range_values, 1e-6, None))
    elevation_deg = np.rad2deg(elevation)

    # Filter by elevation range
    epsilon = 0.5
    min_angle = elevation_angles_deg.min() - epsilon
    max_angle = elevation_angles_deg.max() + epsilon
    valid_elev = (elevation_deg >= min_angle) & (elevation_deg <= max_angle)
    valid = valid_range & valid_elev

    elevation_deg_v = elevation_deg[valid]
    col_v = col_idx[valid]
    range_v = range_values[valid]

    # Find closest row for each point
    row_v = np.argmin(np.abs(elevation_deg_v[:, None] - elevation_angles_deg[None, :]), axis=1)

    # Build range map: sort by range descending, assign so nearest overwrites
    sort_idx = np.argsort(-range_v)
    row_sorted = row_v[sort_idx]
    col_sorted = col_v[sort_idx]
    range_sorted = range_v[sort_idx]

    range_map = np.zeros((n_rows, n_cols), dtype=np.float32)
    range_map[row_sorted, col_sorted] = range_sorted

    return range_map


def get_timestamps_from_dir(input_root, clip_id):
    """Try to load timestamps from the timestamp directory."""
    ts_tar = os.path.join(input_root, 'timestamp', f"{clip_id}.tar")
    if not os.path.exists(ts_tar):
        return None
    try:
        ts_data = get_sample(ts_tar)
        timestamps = {}
        for k, v in ts_data.items():
            if 'timestamp_micros' in k:
                frame_idx = k.split('.')[0]
                timestamps[frame_idx] = int(v)
        return timestamps
    except Exception:
        return None


def process_clip(clip_id, input_root, metadata_output_root, rangemap_output_root,
                 elevation_angles_deg, vehicle_to_lidar, lidar_extrinsic, n_cols=3600):
    """Process a single clip: transform to lidar frame, build range maps, save with correct pose."""
    n_rows = len(elevation_angles_deg)  # always matches elevation table (128)

    lidar_path_tar = os.path.join(input_root, 'lidar_raw', f"{clip_id}.tar")
    if not os.path.exists(lidar_path_tar):
        print(f"Skip: {lidar_path_tar} not found")
        return False

    lidar_data = get_sample(lidar_path_tar)
    if lidar_data is None:
        print(f"Skip: could not load {lidar_path_tar}")
        return False

    # Try to load timestamps
    ts_dict = get_timestamps_from_dir(input_root, clip_id)

    # Get frame indices
    lidar_keys = [k for k in lidar_data.keys() if 'lidar_raw' in k]
    frame_indices = sorted(set(k.split('.')[0] for k in lidar_keys), key=natural_key)

    # Pre-load all frames
    frame_data = {}  # frame_idx -> (xyz_vehicle, vehicle_to_world, timestamp)
    for idx, frame_idx in enumerate(frame_indices):
        lidar_raw_key = f"{frame_idx}.lidar_raw.npz"
        if lidar_raw_key not in lidar_data:
            continue
        lidar_raw = lidar_data[lidar_raw_key]
        xyz_vehicle = lidar_raw['xyz'].astype(np.float32)
        vehicle_to_world = lidar_raw['lidar_to_world'].astype(np.float64)  # actually vehicle_to_world
        if ts_dict and frame_idx in ts_dict:
            timestamp = ts_dict[frame_idx]
        else:
            timestamp = idx * 100000
        frame_data[frame_idx] = (xyz_vehicle, vehicle_to_world, timestamp)

    valid_frame_indices = [f for f in frame_indices if f in frame_data]

    usable_frame_indices = []
    pose_list = []
    timestamps_list = []
    range_map_npz = {'__key__': clip_id}

    rot = vehicle_to_lidar[:3, :3]
    trans = vehicle_to_lidar[:3, 3:4]

    for idx, frame_idx in enumerate(valid_frame_indices):
        xyz_vehicle, vehicle_to_world, current_timestamp = frame_data[frame_idx]

        # Transform xyz: vehicle frame -> LiDAR sensor frame
        xyz_lidar = (rot @ xyz_vehicle.T + trans).T.astype(np.float32)

        # Compose pose: lidar_sensor_to_world = vehicle_to_world @ lidar_extrinsic
        lidar_to_world = vehicle_to_world @ lidar_extrinsic

        # Get next frame pose (for last frame, duplicate current)
        if idx < len(valid_frame_indices) - 1:
            next_frame_idx = valid_frame_indices[idx + 1]
            _, next_vehicle_to_world, next_timestamp = frame_data[next_frame_idx]
            if next_timestamp <= current_timestamp:
                continue
            next_lidar_to_world = next_vehicle_to_world @ lidar_extrinsic
        else:
            # Last frame: duplicate own pose and timestamp + delta
            next_lidar_to_world = lidar_to_world.copy()
            next_timestamp = current_timestamp + 100000

        # Project LiDAR-frame xyz to range map
        range_map = points_to_range_map(xyz_lidar, n_rows, n_cols, elevation_angles_deg)

        # Extract sparse representation
        valid_pixels = np.where(range_map > 0)
        lidar_row = valid_pixels[0]
        lidar_col = valid_pixels[1]
        lidar_range = range_map[valid_pixels]

        range_map_npz[f'{frame_idx}.lidar_row.npz'] = {'arr_0': lidar_row.astype(np.uint8)}
        range_map_npz[f'{frame_idx}.lidar_col.npz'] = {'arr_0': lidar_col.astype(np.uint16)}
        range_map_npz[f'{frame_idx}.lidar_range.npz'] = {'arr_0': lidar_range.astype(np.float16)}

        usable_frame_indices.append(frame_idx)
        pose_list.append([lidar_to_world, next_lidar_to_world])
        timestamps_list.append([current_timestamp, next_timestamp])

    if not usable_frame_indices:
        print(f"Skip: no usable frames for {clip_id}")
        return False

    # Save range maps tar
    tar_path = os.path.join(rangemap_output_root, f"{clip_id}.tar")
    write_to_tar(range_map_npz, tar_path)

    # Save metadata
    os.makedirs(metadata_output_root, exist_ok=True)
    metadata_path = os.path.join(metadata_output_root, f"{clip_id}")
    np.savez(
        metadata_path,
        pose_list=np.array(pose_list, dtype=np.float64),
        timestamps_list=np.array(timestamps_list, dtype=np.int64),
        frame_indices=np.array([int(f) for f in usable_frame_indices], dtype=np.int64),
    )

    return True


def process_clip_worker(args):
    """Multiprocessing wrapper for per-clip conversion."""
    clip_id, input_root, metadata_output_root, rangemap_output_root, elevation_angles_deg, vehicle_to_lidar, lidar_extrinsic, n_cols = args
    try:
        ok = process_clip(
            clip_id,
            input_root,
            metadata_output_root,
            rangemap_output_root,
            elevation_angles_deg,
            vehicle_to_lidar,
            lidar_extrinsic,
            n_cols,
        )
        return clip_id, ok, None
    except Exception as exc:
        return clip_id, False, str(exc)


@click.command()
@click.option("--input_root", type=str, required=True, help="Root of rds_hq_waymo split (e.g. /data2/rds_hq_waymo/training)")
@click.option("--output_root", type=str, required=True, help="Output root (e.g. /data2/rds_hq_waymo/lidar_tokenizer/training)")
@click.option("--split_file", type=str, default=None, help="Optional split file listing clip tar filenames")
@click.option("--n_rows", type=int, default=128, show_default=True, help="Tokenizer-compatible range map height")
@click.option("--n_cols", type=int, default=3600)
@click.option("--num_workers", type=int, default=8, show_default=True, help="Number of clip-level worker processes")
def main(input_root, output_root, split_file, n_rows, n_cols, num_workers):
    metadata_output_root = os.path.join(output_root, "metadata")
    rangemap_output_root = os.path.join(output_root, "lidar")
    os.makedirs(metadata_output_root, exist_ok=True)
    os.makedirs(rangemap_output_root, exist_ok=True)

    if n_rows != 128:
        raise ValueError(f"Waymo tokenizer conversion currently expects n_rows=128, got {n_rows}")

    lidar_extrinsic = WAYMO_TOP_LIDAR_EXTRINSIC
    vehicle_to_lidar = np.linalg.inv(lidar_extrinsic)

    # Create 128 elevation angles from 64 Waymo TOP beams (descending, matching Pandar128)
    elevation_angles_deg = make_elevation_angles(WAYMO_TOP_BEAM_INCLINATIONS_RAD, n_rows)
    print(f"Elevation angles: {len(elevation_angles_deg)} rows, "
          f"range [{elevation_angles_deg.min():.2f}, {elevation_angles_deg.max():.2f}] degrees")

    # Get clip list
    if split_file:
        with open(split_file) as f:
            clip_list = [line.strip().replace('.tar', '') for line in f if line.strip()]
    else:
        lidar_dir = os.path.join(input_root, 'lidar_raw')
        clip_list = sorted([f.replace('.tar', '') for f in os.listdir(lidar_dir) if f.endswith('.tar')])

    print(f"Total clips: {len(clip_list)}")

    success = 0
    failed = []
    num_workers = max(1, num_workers)
    worker_args = [
        (
            clip_id,
            input_root,
            metadata_output_root,
            rangemap_output_root,
            elevation_angles_deg,
            vehicle_to_lidar,
            lidar_extrinsic,
            n_cols,
        )
        for clip_id in clip_list
    ]

    if num_workers == 1:
        for args in tqdm(worker_args, desc="Processing clips"):
            clip_id, ok, error = process_clip_worker(args)
            if ok:
                success += 1
            else:
                if error is not None:
                    print(f"Error processing {clip_id}: {error}")
                failed.append(clip_id)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_clip_worker, args) for args in worker_args]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing clips"):
                clip_id, ok, error = future.result()
                if ok:
                    success += 1
                else:
                    if error is not None:
                        print(f"Error processing {clip_id}: {error}")
                    failed.append(clip_id)

    print(f"\nDone: {success}/{len(clip_list)} clips processed successfully")
    if failed:
        print(f"Failed clips ({len(failed)}): {failed[:10]}...")


if __name__ == "__main__":
    main()
