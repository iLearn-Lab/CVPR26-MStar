from enum import Enum, unique
import torch

@unique
class Node_Type(Enum):
    USER_QUESTION = "USER_QUESTION"
    DIRECT_LOCATION = "DIRECT_LOCATION"
    REPHRASED_OBJECTION_DESCRIPTION = "REPHRASED_OBJECTION_DESCRIPTION"
    BBOX_DESCRIPTION = "BBOX_DESCRIPTION"
    ORIGINAL_IMAGE_DESCRIPTION = "ORIGINAL_IMAGE_DESCRIPTION"
    LOCATION_WITHIN_BBOX = "LOCATION_WITHIN_BBOX"

class GeneratorError(Exception):
    def __init__(self, source, io_input, io_output_list) -> None:
        super().__init__()

        self.source = source
        self.io_input = io_input
        self.io_output_list = io_output_list

def find_last_node_of_type_in_each_branch(root_node, target_type_list):
    last_target_type_nodes = []

    def recursion(node, path_nodes):
        current_path = path_nodes + [node]

        if not node.children:
            last_node_in_path = None
            for n in reversed(current_path):
                if n.node_type in target_type_list:
                    last_node_in_path = n
                    break
            if last_node_in_path and last_node_in_path not in last_target_type_nodes:
                last_target_type_nodes.append(last_node_in_path)
            return

        for child in node.children:
            recursion(child, current_path)

    recursion(root_node, [])
    return last_target_type_nodes

def find_leaf_and_last_target_nodes_in_each_branch(root_node, target_type_list):
    last_target_type_nodes = []
    leaf_nodes = []
    seen_target_nodes = set()

    def recursion(node, path_nodes):
        current_path = path_nodes + [node]

        if not node.children:
            leaf_nodes.append(node)
            last_node_in_path = None

            for n in reversed(current_path):
                if n.node_type in target_type_list:
                    last_node_in_path = n
                    break

            if last_node_in_path and id(last_node_in_path) not in seen_target_nodes:
                last_target_type_nodes.append(last_node_in_path)
                seen_target_nodes.add(id(last_node_in_path))
            return

        for child in node.children:
            recursion(child, current_path)

    assert root_node is not None, "Root node cannot be None"
    recursion(root_node, [])
    return last_target_type_nodes, leaf_nodes

def extract_solution_from_node(node):
    if node.node_type == Node_Type.DIRECT_LOCATION:
        node_answer_bbox = node.direct_bbox
        node_score = node.node_value
    elif node.node_type == Node_Type.LOCATION_WITHIN_BBOX:
        node_answer_bbox = node.bbox_within_the_cropped_image
        node_score = node.node_value
    else:
        node_answer_bbox = None
        node_score = None

    if type(node_answer_bbox) is list and len(node_answer_bbox) == 4:
        return node_answer_bbox, node_score
    else:
        return None, None

def stochastic_find_Grounding_solution(root_node, roolout_node, evaluator):
    all_solution_nodes, all_leaf_nodes = find_leaf_and_last_target_nodes_in_each_branch(root_node, [Node_Type.DIRECT_LOCATION, Node_Type.LOCATION_WITHIN_BBOX])

    if len(all_solution_nodes) == 0:
        return None, None

    valid_solutions = []
    for solution_node in all_solution_nodes:
        bbox_and_score = extract_solution_from_node(solution_node)
        if bbox_and_score is not None:
            bbox, score = bbox_and_score

            valid_solutions.append((bbox, score if score is not None else 1.0))

    if not valid_solutions:
        return None, None, all_solution_nodes, []

    bboxes, scores = zip(*valid_solutions)

    all_solutions = [{'bbox': bbox, 'score': score} for bbox, score in valid_solutions]

    tensor_solutions = torch.tensor(bboxes, dtype=torch.float32)

    merged_bboxes, counts = evaluator._merge_bboxes_with_count(
        tensor_solutions, method="iou_cluster", iou_threshold=0.5
    )

    if len(merged_bboxes) == 0:
        return None, None, all_solution_nodes, all_solutions

    max_count_index = max(enumerate(counts), key=lambda x: x[1])[0]
    merged_score = counts[max_count_index] / len(all_solution_nodes)

    merged_solution = {
        'bbox': merged_bboxes[max_count_index].tolist(),
        'score': merged_score
    }

    rollout_bbox, rollout_score = extract_solution_from_node(roolout_node)
    if rollout_bbox is not None or rollout_score is not None:
        end_solution = {
            'bbox': rollout_bbox,
            'score': rollout_score
        }
    else:
        end_solution = None

    return merged_solution, all_solutions, end_solution, all_solution_nodes, all_leaf_nodes

def convert_bbox_to_absolute(bbox, bbox_type, image_width, image_height):
    if bbox_type == "absolute":
        return bbox
    elif bbox_type == "relative_1000":
        x1 = int(bbox[0] / 1000 * image_width)
        y1 = int(bbox[1] / 1000 * image_height)
        x2 = int(bbox[2] / 1000 * image_width)
        y2 = int(bbox[3] / 1000 * image_height)
        return [x1, y1, x2, y2]
    elif bbox_type == "relative_1":
        x1 = int(bbox[0] * image_width)
        y1 = int(bbox[1] * image_height)
        x2 = int(bbox[2] * image_width)
        y2 = int(bbox[3] * image_height)
        return [x1, y1, x2, y2]
    else:
        raise ValueError(f"Unknown bbox_type: {bbox_type}")
