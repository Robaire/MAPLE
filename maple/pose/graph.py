from numpy.typing import NDArray
from pytransform3d.transformations import concat


class _Node:
    history: list[NDArray]
    attachments: dict[list]  # TODO: Expose this via [] on _Node

    def __init__(self, pose: NDArray):
        """Create a new pose node.
        Args:
            pose: A pytransform transform
        """
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
        self.history.append(pose)

    def get_pose(self):
        """Get the most recent pose in this node's history.
        Returns:
            A pytransform transform
        """
        return self.history[-1]

    def attach(self, key, object):
        """Add an attachment to the node.
        Args:
            key: The lookup key
            object: The item to insert
        """
        try:
            # If key is present, append this object to the list
            self.attachments[key].append(object)
        except KeyError:
            # If key is not present, insert it
            self.attachments[key] = [object]

    def attach_list(self, key, objects):
        """Add a list of attachments to the node.
        Args:
            key: The lookup key
            objects: A list of items to insert
        """
        for object in objects:
            self.attach(key, object)

    def get_attachment(self, key) -> list:
        """Get an attachment list

        This is intended for arbitrary data types that are attached to a pose node for convenience.
        For attachments that are poses use `resolve_attachments` instead.

        Args:
            key: The attachment key
        Returns:
            A list of attachments
        """
        return self.attachments[key]

    def solve_attachment(self, key) -> list[NDArray]:
        """Get an attachment list and automatically transform to the node's reference frame
        Args:
            key: The attachment key
        Returns:
            A list of poses in the node's reference frame
        """

        objects_node = self.get_attachment(key)
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
        return self.nodes[node_key].get_pose()

    def attach(self, attachment_key, object, node_key=None):
        """Add an attachment to the graph.
        Args:
            attachment_key: The lookup key
            object: The item to insert
            node_key: The node key (or None for the most recent node)
        """
        if node_key is not None:
            self.nodes[node_key].attach(attachment_key, object)
        else:
            self.nodes[self.last_node].attach(attachment_key, object)

    def attach_list(self, attachment_key, objects, node_key=None):
        """Add a list of attachments to the graph.
        Args:
            attachment_key: The lookup key
            object: A list of items to insert
            node_key: The node key (or None for the most recent node)
        """
        if node_key is not None:
            self.nodes[node_key].attach_list(attachment_key, objects)
        else:
            self.nodes[self.last_node].attach_list(attachment_key, objects)

    def get_attachment(self, key) -> list:
        """Get an attachment list for all nodes
        Args:
            key: The attachment key
        Returns:
            A list of attachments
        """

        attachments = []
        for node in self.nodes.values():
            try:
                attachments.extend(node.get_attachment(key))
            except KeyError:
                continue

        return attachments

    def solve_attachment(self, key) -> list[NDArray]:
        """Get an attachment list and automatically transform to the graph's reference frame.
        Args:
            key: The attachment key
        Returns:
            A list of poses in the graph's reference frame
        """

        attachments = []
        for node in self.nodes.values():
            try:
                attachments.extend(node.solve_attachment(key))
            except KeyError:
                continue

        return attachments
