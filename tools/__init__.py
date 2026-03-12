"""
Optional tools for 3D representation and scene graph.

Implement get_3d_representation(image) and get_scene_graph(image)
by importing from your tool repository. When not available, returns empty string.
"""


def get_3d_representation(image, object_names=None):
    """Return 3D depth representation (z values, depth ordering) as text."""
    try:
        from tools.depth import extract_3d_representation  # type: ignore
        return extract_3d_representation(image, object_names=object_names)
    except ImportError:
        return ""


def get_scene_graph(image, object_names=None):
    """Return 2D scene graph (nodes, edges) as JSON text."""
    try:
        from tools.scene_graph import extract_scene_graph  # type: ignore
        return extract_scene_graph(image, object_names=object_names)
    except ImportError:
        return ""
