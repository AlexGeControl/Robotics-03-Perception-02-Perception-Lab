#!/usr/bin/env python

# Import PCL module
import pcl

def save(cloud, filename):
    """ Save given pcl using given filename
    """
    pcl.save(cloud, filename)

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

# For table segmentation
cloud = pcl.load_XYZRGB('data/original/tabletop.pcd')

# Voxel Grid filter
voxel_filter = VoxelFilter(
    cloud,
    0.01
)
pcl_clustered = voxel_filter.filter()
save(pcl_clustered, 'data/derived/clustered.pcd')

# PassThrough filter
pass_through_filter = PassThroughFilter(
    pcl_clustered,
    'z',
    [0.6, 1.1]
)
pcl_roi = pass_through_filter.filter()
save(pcl_roi, 'data/derived/roi.pcd')

# RANSAC plane segmentation
plane_segmenter = PlaneSegmenter(
    pcl_roi,
    0.01
)
(idx_table, normal_table) = plane_segmenter.segment()

# Extract table:
pcl_table = pcl_roi.extract(idx_table, negative=False)
save(pcl_table, 'data/derived/table.pcd')

# Extract objects:
pcl_objects = pcl_roi.extract(idx_table, negative=True)
save(pcl_objects, 'data/derived/objects.pcd')

# For outlier removal:
cloud = pcl.load('data/original/tablescene_noisy.pcd')

# Outlier filter:
outlier_filter = OutlierFilter(
    cloud,
    k = 50,
    factor = 1
)
pcl_inliers = outlier_filter.filter()
save(pcl_inliers, 'data/derived/tablescene_denoised.pcd')
