def get_3d_representation(image, object_names=None):
    try:
        from tools.depth import extract_3d_representation  # type: ignore
        return extract_3d_representation(image, object_names=object_names)
    except ImportError:
        return ""


def get_scene_graph(image, object_names=None):
    try:
        from tools.scene_graph import extract_scene_graph  # type: ignore
        return extract_scene_graph(image, object_names=object_names)
    except ImportError:
        return ""
