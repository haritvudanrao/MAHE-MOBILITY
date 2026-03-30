import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import cv2

# Class map:
# 0 = background (sky, buildings, trees)
# 1 = drivable road
# 2 = vehicle (car, truck, bus, motorcycle, bicycle)
# 3 = pedestrian (adult, child, worker, police)
# 4 = barrier (walls, debris, pushable objects)
# 5 = traffic cone

class NuScenesDataset(Dataset):
    def __init__(self, data_root, version='v1.0-mini', img_size=(256, 256)):
        self.data_root = data_root
        self.img_size = img_size

        print("Loading nuScenes...")
        self.nusc = NuScenes(
            version=version,
            dataroot=data_root,
            verbose=False
        )

        print("Loading maps...")
        self.maps = {}
        for loc in ['singapore-queenstown', 'singapore-onenorth',
                    'singapore-hollandvillage', 'boston-seaport']:
            self.maps[loc] = NuScenesMap(
                dataroot=data_root, map_name=loc
            )
        print("All maps loaded!")

        # Collect all valid CAM_FRONT samples
        self.valid_samples = []
        for scene in self.nusc.scene:
            log = self.nusc.get('log', scene['log_token'])
            location = log['location']

            sample_token = scene['first_sample_token']
            while sample_token:
                sample = self.nusc.get('sample', sample_token)

                if 'CAM_FRONT' in sample['data']:
                    cam_token = sample['data']['CAM_FRONT']
                    cam_data = self.nusc.get('sample_data', cam_token)
                    img_path = os.path.join(
                        data_root, cam_data['filename']
                    )

                    if os.path.exists(img_path):
                        self.valid_samples.append({
                            'sample_token': sample['token'],
                            'cam_token': cam_token,
                            'img_path': img_path,
                            'location': location,
                            'anns': sample['anns']
                        })

                sample_token = sample['next']

        print(f"Found {len(self.valid_samples)} valid samples!")

    def _get_camera_info(self, cam_token):
        cam_data = self.nusc.get('sample_data', cam_token)
        cs = self.nusc.get(
            'calibrated_sensor',
            cam_data['calibrated_sensor_token']
        )
        ep = self.nusc.get(
            'ego_pose',
            cam_data['ego_pose_token']
        )
        return cs, ep

    def _get_road_mask(self, sample_info, img_w, img_h):
        cs, ep = self._get_camera_info(sample_info['cam_token'])
        intrinsic = np.array(cs['camera_intrinsic'])
        ego_translation = np.array(ep['translation'])
        ego_rotation = Quaternion(ep['rotation'])

        nusc_map = self.maps[sample_info['location']]
        road_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        try:
            patch_box = (
                ego_translation[0],
                ego_translation[1],
                100, 100
            )

            records = nusc_map.get_records_in_patch(
                patch_box,
                ['drivable_area'],
                mode='intersect'
            )

            for record_token in records['drivable_area']:
                record = nusc_map.get('drivable_area', record_token)

                for polygon_token in record['polygon_tokens']:
                    polygon = nusc_map.extract_polygon(polygon_token)
                    exterior_coords = np.array(polygon.exterior.coords)

                    pts_3d = np.zeros((len(exterior_coords), 3))
                    pts_3d[:, 0] = exterior_coords[:, 0]
                    pts_3d[:, 1] = exterior_coords[:, 1]
                    pts_3d[:, 2] = 0

                    # World to ego
                    pts_ego = pts_3d - ego_translation
                    pts_ego = np.array([
                        ego_rotation.inverse.rotate(p) for p in pts_ego
                    ])

                    # Ego to camera
                    cam_translation = np.array(cs['translation'])
                    cam_rotation = Quaternion(cs['rotation'])
                    pts_cam = pts_ego - cam_translation
                    pts_cam = np.array([
                        cam_rotation.inverse.rotate(p) for p in pts_cam
                    ])

                    # Only points in front of camera
                    valid = pts_cam[:, 2] > 0
                    if valid.sum() < 3:
                        continue

                    pts_cam = pts_cam[valid]

                    pts_img = view_points(
                        pts_cam.T, intrinsic, normalize=True
                    )[:2].T

                    pts_img[:, 0] = np.clip(pts_img[:, 0], 0, img_w-1)
                    pts_img[:, 1] = np.clip(pts_img[:, 1], 0, img_h-1)

                    pts_int = pts_img.astype(np.int32)
                    if len(pts_int) >= 3:
                        cv2.fillPoly(road_mask, [pts_int], 1)

        except Exception:
            road_start = int(img_h * 0.6)
            road_mask[road_start:, :] = 1

        # Road cannot appear in top 40% of image
        road_mask[:int(img_h * 0.4), :] = 0

        return road_mask

    def _get_annotation_masks(self, sample_info, img_w, img_h):
        cs, ep = self._get_camera_info(sample_info['cam_token'])
        intrinsic = np.array(cs['camera_intrinsic'])

        vehicle_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        pedestrian_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        barrier_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cone_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        for ann_token in sample_info['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            category = ann['category_name']

            # Assign to correct mask based on category
            if category in [
                'vehicle.car', 'vehicle.truck', 'vehicle.bus.rigid',
                'vehicle.bus.bendy', 'vehicle.motorcycle',
                'vehicle.bicycle', 'vehicle.trailer',
                'vehicle.construction'
            ]:
                target_mask = vehicle_mask
                cls_val = 2

            elif category in [
                'human.pedestrian.adult', 'human.pedestrian.child',
                'human.pedestrian.construction_worker',
                'human.pedestrian.personal_mobility',
                'human.pedestrian.police_officer'
            ]:
                target_mask = pedestrian_mask
                cls_val = 3

            elif category in [
                'movable_object.barrier',
                'movable_object.debris',
                'movable_object.pushable_pullable'
            ]:
                target_mask = barrier_mask
                cls_val = 4

            elif category == 'movable_object.trafficcone':
                target_mask = cone_mask
                cls_val = 5

            else:
                continue

            box = self.nusc.get_box(ann_token)

            # Transform to ego frame
            box.translate(-np.array(ep['translation']))
            box.rotate(Quaternion(ep['rotation']).inverse)

            # Transform to camera frame
            box.translate(-np.array(cs['translation']))
            box.rotate(Quaternion(cs['rotation']).inverse)

            # Skip boxes behind camera
            if box.center[2] < 0:
                continue

            # Project to image
            corners_2d = view_points(
                box.corners(), intrinsic, normalize=True
            )

            xs = np.clip(corners_2d[0], 0, img_w-1).astype(int)
            ys = np.clip(corners_2d[1], 0, img_h-1).astype(int)

            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()

            if x2 > x1 and y2 > y1:
                # Skip boxes that are too big
                # (means object is too close or projection went wrong)
                box_width = x2 - x1
                box_height = y2 - y1
                
                # Skip if box covers more than 60% of image
                if box_width > img_w * 0.6:
                    continue
                if box_height > img_h * 0.6:
                    continue
                    
                # Skip if box top is above 20% of image
                # (vehicles cant be in the sky!)
                if y1 < img_h * 0.2:
                    continue

                pts = np.array([
                    [x1, y1], [x2, y1],
                    [x2, y2], [x1, y2]
                ], dtype=np.int32)
                cv2.fillPoly(target_mask, [pts], cls_val)
               

        return vehicle_mask, pedestrian_mask, barrier_mask, cone_mask

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample_info = self.valid_samples[idx]

        # Load image at original size
        image = Image.open(sample_info['img_path']).convert('RGB')
        orig_w, orig_h = image.size

        # Generate road mask
        road_mask = self._get_road_mask(sample_info, orig_w, orig_h)

        # Generate annotation masks
        vehicle_mask, pedestrian_mask, barrier_mask, cone_mask = \
            self._get_annotation_masks(sample_info, orig_w, orig_h)

        # Combine all masks
        # Priority: cone > barrier > pedestrian > vehicle > road > background
        final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        final_mask[road_mask == 1] = 1
        final_mask[vehicle_mask == 2] = 2
        final_mask[pedestrian_mask == 3] = 3
        final_mask[barrier_mask == 4] = 4
        final_mask[cone_mask == 5] = 5

        # Resize
        image = image.resize(self.img_size)
        mask_img = Image.fromarray(final_mask)
        mask_img = mask_img.resize(self.img_size, Image.NEAREST)

        # Convert to tensors
        img_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.tensor(img_array).permute(2, 0, 1)
        mask_tensor = torch.tensor(
            np.array(mask_img, dtype=np.int64)
        )

        return image_tensor, mask_tensor