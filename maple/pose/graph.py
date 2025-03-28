from numpy.typing import NDArray
from pytransform3d.transformations import concat, transform_from
from scipy.spatial.transform import Rotation


class _Node:
    history: list[NDArray]
    attachments: dict[list]

    def __init__(self, pose: NDArray):
        """Create a new pose node.
        Args:
            pose: A pytransform transform or None.
            If None, create an empty Node without pose information that can only contain attachments.
        """

        if pose is None:
            self.history = None
        else:
            self.history = [pose]

        self.attachments = {}

    def __str__(self):
        # TODO: Implement
        pass

    def update_pose(self, pose: NDArray):
        """Updates this node's pose.
        Args:
            pose: A pytransform transform
        """

        if self.history is None:
            raise ValueError("Cannot update pose of an empty node.")
        else:
            self.history.append(pose)

    def get_pose(self):
        """Get the most recent pose in this node's history.
        Returns:
            A pytransform transform
        """

        if self.history is None:
            raise ValueError("Cannot get the pose of an empty node.")

        else:
            return self.history[-1]

    def __getitem__(self, key):
        """Get an attachment list

        This is intended for arbitrary data types that are attached to a pose node for convenience.
        For attachments that are poses use `resolve_attachments` instead.
        """
        # If the key does not exist in the dict, create it.
        # This enables Node[key] += [a, b, c, ...] without setting Node[key] = [] ahead of time
        try:
            return self.attachments[key]
        except KeyError:
            self.attachments[key] = []
            return self.attachments[key]

    def __setitem__(self, key, value):
        self.attachments[key] = value

    def __delitem__(self, key):
        del self.attachments[key]

    def solve_attachment(self, key) -> list[NDArray]:
        """Get an attachment list and automatically transform to the node's reference frame
        Args:
            key: The attachment key
        Returns:
            A list of poses in the node's reference frame
        """

        if self.history is None:
            raise ValueError("Cannot solve attachment of an empty node.")

        else:
            objects_node = self[key]
            node_reference = self.get_pose()
            return [concat(object_node, node_reference) for object_node in objects_node]


class PoseGraph:
    nodes: dict[_Node]
    sequential: bool
    last_node: int

    def __init__(self, sequential=True):
        """
        Args:
            sequential: If the pose graph should automatically generate keyframe index
        """
        self.nodes = {}
        self.sequential = sequential
        self.last_node = -1

    def __str__(self):
        # TODO: Implement
        pass

    def add_node(self, pose: NDArray, key=None):
        """Add a node to the graph.
        Args:
            pose: The pose to add
            key: A key to use for the node (must be unique and sortable)

        Returns:
            The node key
        """

        node = _Node(pose)

        if self.sequential:
            self.last_node += 1
            self.nodes[self.last_node] = node

        elif key is not None:
            self.nodes[key] = node
            self.last_node = key

        else:
            raise ValueError("A key must be provided for a non-sequential PoseGraph")

        return self.last_node

    def update_pose(self, node_key, pose: NDArray):
        """Update a node on the graph.
        Args:
            key: The key for this node
            pose: The new pose
        """

        self.nodes[node_key].update_pose(pose)

    def get_pose(self, node_key) -> NDArray:
        # TODO: If a node is empty we need to interpolate the pose if possible
        # Does this require simulation time? Or should we assume nodes are added at a fixed interval?
        # If the latter we could weight the interpolation based on the distance to the neighbor nodes with pose data

        try:
            return self.nodes[node_key].get_pose()

        # The node does not have a pose associated with it
        except ValueError:
            # Get the list of all node keys
            node_keys = self.nodes.keys()
            index = node_keys.index(node_key)

            # Search backwards for the previous node
            dist_previous = 0
            previous_pose = None
            for node_key in reversed(node_keys[:index]):
                dist_previous += 1

                try:
                    previous_pose = self.nodes[node_key].get_pose()
                    break
                except ValueError:
                    continue

            # Search forwards for the next node
            dist_next = 0
            next_pose = None
            for node_key in node_keys[index:]:
                dist_next += 1

                try:
                    next_pose = self.nodes[node_key].get_pose()
                    break
                except ValueError:
                    continue

            if previous_pose is None or next_pose is None:
                raise ValueError("Cannot interpolate node pose.")

            # Interpolate a pose
            alpha = dist_previous / (dist_previous + dist_next)

            previous_q = Rotation.from_matrix(previous_pose[:3, :3]).as_quat()
            previous_t = previous_pose[:3, 3]

            next_q = Rotation.from_matrix(next_pose[:3, :3]).as_quat()
            next_t = next_pose[:3, 3]

            interp_q = Rotation.slerp(alpha, Rotation.from_quat([previous_q, next_q]))
            interp_t = (1 - alpha) * previous_t + alpha * next_t

            return transform_from(interp_q.as_matrix(), interp_t)

    def attach(self, attachment_key, object, node_key=None):
        """Add an attachment to the graph.
        Args:
            attachment_key: The lookup key
            object: The item to insert
            node_key: The node key (or None for the most recent node)
        """
        if node_key is not None:
            self.nodes[node_key][attachment_key] += [object]
        else:
            self.nodes[self.last_node][attachment_key] += [object]

    def attach_list(self, attachment_key, objects, node_key=None):
        """Add a list of attachments to the graph.
        Args:
            attachment_key: The lookup key
            object: A list of items to insert
            node_key: The node key (or None for the most recent node)
        """
        if node_key is not None:
            self.nodes[node_key][attachment_key] += objects
        else:
            self.nodes[self.last_node][attachment_key] += objects

    def get_attachment(self, key):
        """Get an attachment list for all nodes
        Args:
            key: The attachment key
        Returns:
            A list of attachments
        """
        return self[key]

    def __getitem__(self, key):
        """Get an attachment list for all nodes
        Args:
            key: The attachment key
        Returns:
            A list of attachments
        """
        attachments = []
        for node in self.nodes.values():
            attachments.extend(node[key])

        return attachments

    def solve_attachment(self, key) -> list[NDArray]:
        """Get an attachment list and automatically transform to the graph's reference frame.
        Args:
            key: The attachment key
        Returns:
            A list of poses in the graph's reference frame. Empty nodes are estimated
        """

        attachments = []

        for node_key, node in self.nodes.items():
            try:
                objects_node = node[key]
                node_reference = self.get_pose(
                    node_key
                )  # TODO: What if this cant be solved?
                attachments.extend(
                    [
                        concat(object_node, node_reference)
                        for object_node in objects_node
                    ]
                )
            except KeyError:
                continue

        return attachments