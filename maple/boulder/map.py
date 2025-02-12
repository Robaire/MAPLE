import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN


class BoulderMap:
    def __init__(self, geometric_map):
        """
        Args:
            geometric_map: The GeometricMap object from the leaderboard
        """

        self.geometric_map = geometric_map

    def _generate_map(self, boulders_global: list) -> NDArray:
        """Generates a 2D array for the boulder locations in the map.
        Args:
            boulders_global: A list of transforms representing centroids of
                boulder detections in the global frame

        Returns:
            A 2D boolean array representing the locations of boulders in the map
        """

        size = self.geometric_map.get_cell_number()
        # size = self.geometric_map.get_map_size()
        # size = int(np.ceil(size / 0.15))
        print(f"Size: {size}")
        boulder_map = np.zeros((size, size), dtype=bool)

        # Extract x,y coordinates from transforms
        points = np.array([boulder[:3, 3][:2] for boulder in boulders_global])

        if len(points) == 0:
            return boulder_map

        # Run DBSCAN
        # eps = 0.15 (grid size) - points closer than this are considered neighbors
        # min_samples = 2 - require at least 2 points to form a cluster
        clustering = DBSCAN(eps=0.15, min_samples=2).fit(points)

        # Get cluster labels (-1 is noise)
        labels = clustering.labels_

        # Process each cluster (including noise points)
        unique_labels = set(labels)
        for label in unique_labels:
            cluster_points = points[labels == label]

            # Ignore noise points
            if label != -1:
                # Use mean position of cluster
                cluster_center = np.mean(cluster_points, axis=0)
                points_to_mark = [cluster_center]

            # Mark cells for each point
            for point in points_to_mark:
                # Convert world coordinates to grid cell indices
                cell_indices = self.geometric_map.get_cell_indexes(point[0], point[1])
                if cell_indices[0] is not None and cell_indices[1] is not None:
                    if 0 <= cell_indices[0] < size and 0 <= cell_indices[1] < size:
                        boulder_map[cell_indices] = True

        """
        Notes:

        `boulders_global` is a point cloud representing a potential point on the surface
        of a boulder. However, the system generating these points isn't perfect.
        There will be extraneous outliers that are not actually boulders but 
        artifacts that were misinterpreted as boulders. In theory, real boulders
        should have been detected multiple times, therefore we should expect to
        see clusters of points where real boulders are. We need to filter the 
        point cloud to look for clusters and log these as "real boulders" so to
        speak.
        """

        return boulder_map

    def set_map(self, samples: list):
        """Set the boulder locations in the geometric_map
        Args:
            samples: A list of boulder location sample points
        """

        boulder_map = self._generate_map(samples)

        for x, y in np.ndindex(boulder_map.shape):
            self.geometric_map.set_cell_rock(x, y, boulder_map[x, y])
