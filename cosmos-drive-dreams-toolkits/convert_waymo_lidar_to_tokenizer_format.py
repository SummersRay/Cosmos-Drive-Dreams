"""
Convert Waymo rds_hq lidar_raw data to LiDAR tokenizer training format.

Waymo rds_hq lidar_raw contains only 'xyz' (vehicle frame) and 'lidar_to_world'
(vehicle_to_world). This script projects xyz to spherical coordinates to build
128x3600 range maps, matching the format expected by the tokenizer.

No ncore dependency required.

Output format:
  metadata/{clip_id}.npz  - pose_list, timestamps_list, frame_indices
  lidar/{clip_id}.tar     - sparse range maps (row/col/range/intensity per frame)

Usage:
  python convert_waymo_lidar_to_tokenizer_format.py \
      --input_root /data2/rds_hq_waymo/training \
      --output_root /data2/rds_hq_waymo/lidar_tokenizer/training
"""

import os
import json
import re
import click
import numpy as np
from pathlib import Path
from tqdm import tqdm
from webdataset import WebDataset, non_empty, TarWriter


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


def load_sensor_elevation_angles(param_path="assets/row-offset-spinning-lidar-model-parameters.json"):
    """Load elevation angles from the default sensor model parameters."""
    param = json.load(open(param_path, "r"))
    return np.rad2deg(np.array(param['row_elevations_rad']))  # in degrees


def make_uniform_elevation_angles(n_rows, fov_min=-3.0, fov_max=20.0):
    """Create uniformly spaced elevation angles for Waymo LiDAR.

    Waymo's merged point cloud (top + side LiDARs) covers roughly [-3, 20] degrees
    in elevation. Using uniform spacing avoids the sparse ring problem caused by
    mismatched sensor models.
    """
    return np.linspace(fov_min, fov_max, n_rows)


def points_to_range_map(xyz, n_rows, n_cols, sensor_elevation_angles, max_range=105):
    """
    Project xyz points to a range map using spherical coordinates.
    Uses scatter-min to keep nearest point per pixel (via sort + assign).

    Args:
        xyz: (N, 3) float32 array, points in sensor/vehicle frame
        n_rows: number of rows (elevation bins)
        n_cols: number of columns (azimuth bins)
        sensor_elevation_angles: (n_rows,) array of elevation angles in degrees
        max_range: maximum range threshold

    Returns:
        range_map: (n_rows, n_cols) float32
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    range_values = np.sqrt(x**2 + y**2 + z**2)

    # Filter by range
    valid_range = (range_values > 0) & (range_values < max_range)

    # Compute azimuth -> column index
    azimuth = -np.arctan2(y, x) + np.pi  # [0, 2*pi)
    col_idx = ((azimuth / (2 * np.pi)) * n_cols).astype(np.int32)
    col_idx = np.clip(col_idx, 0, n_cols - 1)

    # Compute elevation -> row index (find nearest elevation angle)
    elevation = np.arcsin(z / np.clip(range_values, 1e-6, None))
    elevation_deg = np.rad2deg(elevation)

    # Filter by elevation range
    epsilon = 0.5
    min_angle = sensor_elevation_angles.min() - epsilon
    max_angle = sensor_elevation_angles.max() + epsilon
    valid_elev = (elevation_deg >= min_angle) & (elevation_deg <= max_angle)
    valid = valid_range & valid_elev

    elevation_deg_v = elevation_deg[valid]
    col_v = col_idx[valid]
    range_v = range_values[valid]

    # Find closest row for each point
    row_v = np.argmin(np.abs(elevation_deg_v[:, None] - sensor_elevation_angles[None, :]), axis=1)

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
                 sensor_elevation_angles, n_rows=128, n_cols=3600):
    """Process a single clip: build range maps from raw lidar xyz and save."""
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

    usable_frame_indices = []
    pose_list = []
    timestamps_list = []
    range_map_npz = {'__key__': clip_id}

    for idx, frame_idx in enumerate(frame_indices):
        lidar_raw_key = f"{frame_idx}.lidar_raw.npz"
        if lidar_raw_key not in lidar_data:
            continue

        lidar_raw = lidar_data[lidar_raw_key]
        xyz = lidar_raw['xyz'].astype(np.float32)
        ego_pose = lidar_raw['lidar_to_world'].astype(np.float32)

        # Get timestamp: use real timestamps if available, otherwise synthesize from
        # sequential index (idx). Waymo runs at 10Hz so each frame is 100ms apart.
        # Note: frame_idx may be non-sequential (e.g. 0,3,6,...) so we use idx instead.
        if ts_dict and frame_idx in ts_dict:
            current_timestamp = ts_dict[frame_idx]
        else:
            current_timestamp = idx * 100000  # 100ms per frame at 10Hz, in microseconds

        # Need next frame for pose pair (last frame is intentionally excluded because
        # there is no next-frame pose available for motion compensation)
        if idx < len(frame_indices) - 1:
            next_frame_idx = frame_indices[idx + 1]
            next_key = f"{next_frame_idx}.lidar_raw.npz"
            if next_key not in lidar_data:
                continue
            next_raw = lidar_data[next_key]
            next_pose = next_raw['lidar_to_world'].astype(np.float32)
            if ts_dict and next_frame_idx in ts_dict:
                next_timestamp = ts_dict[next_frame_idx]
            else:
                next_timestamp = (idx + 1) * 100000
            if next_timestamp <= current_timestamp:
                continue
        else:
            continue

        # Project xyz to range map
        range_map = points_to_range_map(xyz, n_rows, n_cols, sensor_elevation_angles)

        # Extract sparse representation
        valid_pixels = np.where(range_map > 0)
        lidar_row = valid_pixels[0]
        lidar_col = valid_pixels[1]
        lidar_range = range_map[valid_pixels]

        range_map_npz[f'{frame_idx}.lidar_row.npz'] = {'arr_0': lidar_row.astype(np.uint8)}
        range_map_npz[f'{frame_idx}.lidar_col.npz'] = {'arr_0': lidar_col.astype(np.uint16)}
        range_map_npz[f'{frame_idx}.lidar_range.npz'] = {'arr_0': lidar_range.astype(np.float16)}

        usable_frame_indices.append(frame_idx)
        pose_list.append([ego_pose, next_pose])
        timestamps_list.append([current_timestamp, next_timestamp])

    if not usable_frame_indices:
        print(f"Skip: no usable frames for {clip_id}")
        return False

    # Save range maps tar
    tar_path = os.path.join(rangemap_output_root, f"{clip_id}.tar")
    write_to_tar(range_map_npz, tar_path)

    # Save metadata with proper dtypes (avoid object arrays that need allow_pickle)
    os.makedirs(metadata_output_root, exist_ok=True)
    metadata_path = os.path.join(metadata_output_root, f"{clip_id}")
    np.savez(
        metadata_path,
        pose_list=np.array(pose_list, dtype=np.float64),
        timestamps_list=np.array(timestamps_list, dtype=np.int64),
        frame_indices=np.array([int(f) for f in usable_frame_indices], dtype=np.int64),
    )

    return True


@click.command()
@click.option("--input_root", type=str, required=True, help="Root of rds_hq_waymo split (e.g. /data2/rds_hq_waymo/training)")
@click.option("--output_root", type=str, required=True, help="Output root (e.g. /data2/rds_hq_waymo/lidar_tokenizer/training)")
@click.option("--split_file", type=str, default=None, help="Optional split file listing clip tar filenames")
@click.option("--sensor_params", type=str, default=None, help="Sensor parameter JSON file (default: use uniform FOV for Waymo)")
@click.option("--fov_min", type=float, default=-3.0, help="Min elevation angle in degrees (for uniform mode)")
@click.option("--fov_max", type=float, default=20.0, help="Max elevation angle in degrees (for uniform mode)")
@click.option("--n_rows", type=int, default=128)
@click.option("--n_cols", type=int, default=3600)
def main(input_root, output_root, split_file, sensor_params, fov_min, fov_max, n_rows, n_cols):
    metadata_output_root = os.path.join(output_root, "metadata")
    rangemap_output_root = os.path.join(output_root, "lidar")
    os.makedirs(metadata_output_root, exist_ok=True)
    os.makedirs(rangemap_output_root, exist_ok=True)

    # Load or generate sensor elevation angles
    if sensor_params:
        sensor_elevation_angles = load_sensor_elevation_angles(sensor_params)
        if len(sensor_elevation_angles) != n_rows:
            raise ValueError(
                f"Sensor params have {len(sensor_elevation_angles)} elevation angles "
                f"but n_rows={n_rows}. These must match to avoid index errors."
            )
    else:
        sensor_elevation_angles = make_uniform_elevation_angles(n_rows, fov_min, fov_max)
    print(f"Sensor elevation angles: {len(sensor_elevation_angles)} rows, "
          f"range [{sensor_elevation_angles.min():.1f}, {sensor_elevation_angles.max():.1f}] degrees")

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
    for clip_id in tqdm(clip_list, desc="Processing clips"):
        try:
            if process_clip(clip_id, input_root, metadata_output_root, rangemap_output_root,
                           sensor_elevation_angles, n_rows, n_cols):
                success += 1
            else:
                failed.append(clip_id)
        except Exception as e:
            print(f"Error processing {clip_id}: {e}")
            failed.append(clip_id)

    print(f"\nDone: {success}/{len(clip_list)} clips processed successfully")
    if failed:
        print(f"Failed clips ({len(failed)}): {failed[:10]}...")


if __name__ == "__main__":
    main()
