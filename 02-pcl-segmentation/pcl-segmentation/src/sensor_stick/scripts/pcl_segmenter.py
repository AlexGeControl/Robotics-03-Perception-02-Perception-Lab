#!/usr/bin/env python

# Import modules:
from pcl_helper import *

# PCLSegmenter:
class PCLSegmenter():
    """ Segment PCL using DBSCAN
    """
    # Filter utilities:
    class VoxelFilter():
        """ Voxel filter for point cloud
        """
        def __init__(self, cloud, leaf_size = 0.0618):
            """ Instantiate voxel filter

            Args:
                    cloud: pcl.cloud, input point cloud
                leaf_size: voxel dimension

            Returns:

            """
            self._leaf_size = leaf_size

            self._filter = cloud.make_voxel_grid_filter()
            self._filter.set_leaf_size(
                *([self._leaf_size]*3)
            )

        def filter(self):
            """ Filter input point cloud
            """
            return self._filter.filter()

    class PassThroughFilter():
        """ Pass through filter for spatial ROI selection
        """
        def __init__(
            self,
            cloud,
            name,
            limits
        ):
            """ Instantiate pass through filter

            Args:
                    cloud: pcl.cloud, input point cloud
                     name: filter field name
                   limits: filter field value range, in (min, max) format
            """
            self._name = name
            self._limits = limits

            self._filter = cloud.make_passthrough_filter()
            self._filter.set_filter_field_name(
                self._name
            )
            self._filter.set_filter_limits(
                *self._limits
            )

        def filter(self):
            """ Filter input point cloud
            """
            return self._filter.filter()

    class OutlierFilter():
        """ Remove outliers in PCL
        """
        def __init__(self, cloud, k = 50, factor = 1):
            """ Instantiate statistical outlier filter
            """
            self._k = k
            self._factor = factor

            self._filter = cloud.make_statistical_outlier_filter()
            # Set the number of neighboring points to analyze for any given point
            self._filter.set_mean_k(self._k)
            # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
            self._filter.set_std_dev_mul_thresh(self._factor)

        def filter(self):
            """ Return inliers
            """
            return self._filter.filter()

    class PlaneSegmenter():
        """ Dominant plane segmenter for PCL
        """
        def __init__(self, cloud, max_distance = 1):
            """ Instantiate plane segmenter
            """
            self._max_distance = max_distance

            self._segmenter = cloud.make_segmenter()
            self._segmenter.set_model_type(pcl.SACMODEL_PLANE)
            self._segmenter.set_method_type(pcl.SAC_RANSAC)
            self._segmenter.set_distance_threshold(self._max_distance)

        def segment(self):
            """ Segment the dominant plane
            """
            return self._segmenter.segment()

    class EuclideanSegmenter():
        """ Segment PCL using DBSCAN
        """
        def __init__(
            self,
            cloud,
            eps = 0.001, min_samples = 10, max_samples = 250
        ):
            """ Instantiate Euclidean segmenter
            """
            # 1. Convert XYZRGB to XYZ:
            self._cloud = XYZRGB_to_XYZ(cloud)
            self._tree = self._cloud.make_kdtree()

            # 2. Set params:
            self._eps = eps
            self._min_samples = min_samples
            self._max_samples = max_samples

            # 3. Create segmenter:
            self._segmenter = self._cloud.make_EuclideanClusterExtraction()
            self._segmenter.set_ClusterTolerance(self._eps)
            self._segmenter.set_MinClusterSize(self._min_samples)
            self._segmenter.set_MaxClusterSize(self._max_samples)
            self._segmenter.set_SearchMethod(self._tree)

        def segment(self):
            """ Segment objects
            """
            # 1. Identify clusters and their colors:
            cluster_indices = self._segmenter.Extract()
            cluster_color = get_color_list(
                len(cluster_indices)
            )

            # 2. Add color:
            XYZRGB = []
            for cluster_id, indices in enumerate(cluster_indices):
                for indice in indices:
                    XYZRGB.append(
                        [
                            self._cloud[indice][0],
                            self._cloud[indice][1],
                            self._cloud[indice][2],
                            rgb_to_float(
                                cluster_color[cluster_id]
                            )
                        ]
                    )

            # 3. Format as XYZRGB cloud:
            cluster_cloud = pcl.PointCloud_PointXYZRGB()
            cluster_cloud.from_list(XYZRGB)

            return cluster_cloud

    def __init__(self):
        # Initialize ROS node:
        rospy.init_node('pcl_segmenter')

        # Create Subscribers
        self._sub_pcl = rospy.Subscriber(
            '/sensor_stick/point_cloud',
            PointCloud2,
            self._handle_pcl,
            queue_size=10
        )

        # Create Publishers
        self._pub_pcl_table = rospy.Publisher(
            '/pcl_table',
            PointCloud2,
            queue_size=10
        )
        self._pub_pcl_objects = rospy.Publisher(
            '/pcl_objects',
            PointCloud2,
            queue_size=10
        )
        self._pub_pcl_separate_objects = rospy.Publisher(
            '/pcl_cluster',
            PointCloud2,
            queue_size=10
        )

        # Spin till shutdown:
        while not rospy.is_shutdown():
            rospy.spin()

    # Callback function for your Point Cloud Subscriber
    def _handle_pcl(self, ros_cloud):
        """ Handle ROS pc2 message
        """
        # Convert ROS msg to PCL data
        pcl_original = ros_to_pcl(ros_cloud)

        # 1. Voxel grid downsampling
        voxel_filter = PCLSegmenter.VoxelFilter(
            pcl_original,
            0.01
        )
        pcl_clustered = voxel_filter.filter()

        # 2. PassThrough filter
        pass_through_filter = PCLSegmenter.PassThroughFilter(
            pcl_clustered,
            'z',
            [0.6, 1.1]
        )
        pcl_roi = pass_through_filter.filter()

        # 3. RANSAC plane segmentation
        plane_segmenter = PCLSegmenter.PlaneSegmenter(
            pcl_roi,
            0.01
        )
        (idx_table, normal_table) = plane_segmenter.segment()

        # 4. Extract table & objects:
        pcl_table = pcl_roi.extract(idx_table, negative=False)
        pcl_objects = pcl_roi.extract(idx_table, negative=True)

        ros_cloud_table = pcl_to_ros(pcl_table)
        ros_cloud_objects = pcl_to_ros(pcl_objects)

        self._pub_pcl_table.publish(ros_cloud_table)
        self._pub_pcl_objects.publish(ros_cloud_objects)

        # 6. Extract seperate objects using DBSCAN
        object_segmenter = PCLSegmenter.EuclideanSegmenter(
            pcl_objects,
            eps = 0.025, min_samples = 16, max_samples = 2048
        )
        pcl_separate_objects = object_segmenter.segment()

        ros_cloud_separate_objects = pcl_to_ros(pcl_separate_objects)
        self._pub_pcl_separate_objects.publish(ros_cloud_separate_objects)

if __name__ == '__main__':
    try:
        # Initialize color list:
        get_color_list.color_list = []

        PCLSegmenter()
    except ROSInterruptException:
        pass
