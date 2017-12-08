#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

import pickle

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

class PCLClassifier():
    """ Segmented PCL classifier
    """
    # Classifier utilities:
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
            # 1. Segment objects:
            cluster_indices = self._segmenter.Extract()

            # 2. Generate positions for markers:
            cluster_reps = []
            for idx_points in cluster_indices:
                rep_position = list(self._cloud[idx_points[0]])
                rep_position[2] += 0.4

                cluster_reps.append(rep_position)

            return (cluster_indices, cluster_reps)

    def __init__(self, model_filename):
        # Load trained classifier:
        model = pickle.load(
            open(model_filename, 'rb')
        )

        # 1. Classifier:
        self._clf = model['classifier']
        # 2. Label encoder:
        self._encoder = LabelEncoder()
        self._encoder.classes_ = model['classes']
        # 3. Feature scaler:
        self._scaler = model['scaler']

        # Initialize ros node:
        rospy.init_node('pcl_classifier')

        # Create Subscribers
        self._sub_pcl = rospy.Subscriber(
            '/sensor_stick/point_cloud',
            PointCloud2,
            self._handle_pcl_classification,
            queue_size=10
        )

        # Create Publishers
        self._pub_pcl_objects = rospy.Publisher(
            '/detected_objects',
            DetectedObjectsArray,
            queue_size=10
        )
        self._pub_pcl_classes = rospy.Publisher(
            '/object_markers',
            Marker,
            queue_size=10
        )

        # Spin till shutdown:
        while not rospy.is_shutdown():
            rospy.spin()

    def _handle_pcl_classification(self, ros_cloud):
        """ Handle ROS pc2 message
        """
        # Convert ROS msg to PCL data
        pcl_original = ros_to_pcl(ros_cloud)

        # 1. Voxel grid downsampling
        voxel_filter = PCLClassifier.VoxelFilter(
            pcl_original,
            0.01
        )
        pcl_clustered = voxel_filter.filter()

        # 2. PassThrough filter
        pass_through_filter = PCLClassifier.PassThroughFilter(
            pcl_clustered,
            'z',
            [0.6, 1.1]
        )
        pcl_roi = pass_through_filter.filter()

        # 3. RANSAC plane segmentation
        plane_segmenter = PCLClassifier.PlaneSegmenter(
            pcl_roi,
            0.01
        )
        (idx_table, normal_table) = plane_segmenter.segment()

        # 4. Extract objects:
        pcl_objects = pcl_roi.extract(idx_table, negative=True)

        # 5. Extract seperate objects using DBSCAN
        object_segmenter = PCLClassifier.EuclideanSegmenter(
            pcl_objects,
            eps = 0.025, min_samples = 16, max_samples = 2048
        )
        (cluster_indices, cluster_reps) = object_segmenter.segment()

        detected_objects = []
        detected_object_labels = []
        for idx_object, idx_points in enumerate(cluster_indices):
            # Grab the points for the cluster from the extracted outliers (cloud_objects)
            pcl_object = pcl_objects.extract(idx_points)

            # Convert the cluster from pcl to ROS using helper function
            ros_cloud_object = pcl_to_ros(pcl_object)

            # Extract histogram features
            color_hist = compute_color_histograms(ros_cloud_object, using_hsv=True)
            normal_hist = compute_normal_histograms(get_normals(ros_cloud_object))
            feature = np.concatenate(
                (color_hist, normal_hist)
            )

            # Make the prediction
            prediction = self._clf.predict(
                self._scaler.transform(
                    feature.reshape(1,-1)
                )
            )
            label = self._encoder.inverse_transform(
                prediction
            )[0]

            # Add the detected object to the list of detected objects.
            detected_object = DetectedObject()
            detected_object.label = label
            detected_object.cloud = pcl_object
            detected_objects.append(
                detected_object
            )
            detected_object_labels.append(label)
            # Publish object label into RViz
            self._pub_pcl_classes.publish(
                make_label(label,cluster_reps[idx_object], idx_object)
            )

        rospy.loginfo(
            'Detected {} objects: {}'.format(
                len(detected_object_labels),
                detected_object_labels
            )
        )
        self._pub_pcl_objects.publish(detected_objects)

if __name__ == '__main__':
    try:
        # Initialize color_list
        get_color_list.color_list = []

        PCLClassifier('model.sav')
    except rospy.ROSInterruptException:
        pass
