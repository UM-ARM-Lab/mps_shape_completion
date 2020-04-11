#!/usr/bin/env python3

import argparse

import numpy as np

import rospy
from mps_shape_completion_msgs.msg import OccupancyStamped
from mps_shape_completion_visualization import conversions


def to_msg(voxel_grid):
    return conversions.vox_to_occupancy_stamped(voxel_grid,
                                                dim=voxel_grid.shape[1],
                                                scale=0.01,
                                                frame_id="object")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npz_file', help='*.npz file with scale, known_occ, gt_occ, known_free. grids are assumed to be 64x64x64')

    args = parser.parse_args()

    data = np.load(args.npz_file)

    rospy.init_node('publish_npy_occupancy_stamped')
    gt_occ_pub = rospy.Publisher('gt_voxel_grid', OccupancyStamped, queue_size=10)
    known_occ_pub = rospy.Publisher('known_occ', OccupancyStamped, queue_size=10)
    known_free_pub = rospy.Publisher('known_free', OccupancyStamped, queue_size=10)

    gt_occ_msg = to_msg(data['gt_occ'])
    known_occ_msg = to_msg(data['known_occ'])
    if 'known_free' in data:
        known_free_msg = to_msg(data['known_free'])

    for _ in range(4):
        rospy.sleep(0.1)
        gt_occ_pub.publish(gt_occ_msg)
        known_occ_pub.publish(known_occ_msg)
        if 'known_free' in data:
            known_free_pub.publish(known_free_msg)


if __name__ == '__main__':
    main()
