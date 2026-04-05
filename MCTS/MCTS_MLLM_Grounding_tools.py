from run_src.mstar_Grounding_utils import (
    Node_Type,
)
from typing import List, Dict, Tuple, Any
from copy import deepcopy

def node_initial_state_check(parent, depth, node_type, node_value, generator, user_question, raw_object_descriptions, object_descriptions, question_image, max_depth_allowed, wrong_terminal_node_flag, expected_answer, direct_bbox, record_bbox, descriptions_within_bbox, target_in_bbox_flag, original_image_description, target_existence_flag, bbox_within_the_cropped_image, is_new_cropped, rephrased_objection_descriptions):
    try:
        assert depth is not None
        assert node_type is not None

        if parent is not None:
            assert node_value is not None

        if node_type is Node_Type.USER_QUESTION:
            assert depth == 0
            assert all(
                attr is None
                for attr in [
                    parent,
                    node_value,
                    direct_bbox,
                    record_bbox,
                    descriptions_within_bbox,
                    target_in_bbox_flag,
                    original_image_description,
                    bbox_within_the_cropped_image,
                    is_new_cropped,

                    rephrased_objection_descriptions,
                ]
            )
            assert all(
                attr is not None
                for attr in [generator, expected_answer, user_question, raw_object_descriptions, max_depth_allowed]
            )
        elif node_type is Node_Type.DIRECT_LOCATION:
            assert depth > 0
            assert all(
                attr is None
                for attr in [
                    generator,
                    user_question,
                    expected_answer,
                    max_depth_allowed,
                    descriptions_within_bbox,
                    target_in_bbox_flag,
                    original_image_description,
                    bbox_within_the_cropped_image,

                    rephrased_objection_descriptions,
                ]
            )
            assert all(attr is not None for attr in [parent, node_value, direct_bbox, record_bbox, wrong_terminal_node_flag, is_new_cropped])
        elif node_type is Node_Type.BBOX_DESCRIPTION:
            assert depth > 0
            assert all(
                attr is None
                for attr in [
                    generator,
                    user_question,
                    expected_answer,
                    max_depth_allowed,
                    direct_bbox,
                    record_bbox,

                    original_image_description,
                    bbox_within_the_cropped_image,
                    is_new_cropped,

                    rephrased_objection_descriptions,
                ]
            )
            assert all(
                attr is not None for attr in [parent, node_value, descriptions_within_bbox, target_in_bbox_flag]
            )
        elif node_type is Node_Type.ORIGINAL_IMAGE_DESCRIPTION:
            assert depth == 1
            assert all(
                attr is None
                for attr in [
                    generator,
                    user_question,
                    expected_answer,
                    max_depth_allowed,
                    direct_bbox,
                    record_bbox,
                    descriptions_within_bbox,
                    target_in_bbox_flag,

                    bbox_within_the_cropped_image,
                    is_new_cropped,

                    rephrased_objection_descriptions,
                ]
            )
            assert all(attr is not None for attr in [parent, node_value, original_image_description, target_existence_flag])
        elif node_type is Node_Type.LOCATION_WITHIN_BBOX:
            assert depth > 0
            assert all(
                attr is None
                for attr in [
                    generator,
                    user_question,
                    expected_answer,
                    max_depth_allowed,
                    direct_bbox,

                    descriptions_within_bbox,
                    target_in_bbox_flag,
                    original_image_description,

                    rephrased_objection_descriptions,
                ]
            )
            assert all(attr is not None for attr in [parent, node_value, record_bbox, bbox_within_the_cropped_image, is_new_cropped])
        elif node_type is Node_Type.REPHRASED_OBJECTION_DESCRIPTION:
            assert depth > 0
            assert all(
                attr is None
                for attr in [
                    generator,
                    user_question,
                    expected_answer,
                    max_depth_allowed,
                    direct_bbox,
                    record_bbox,
                    descriptions_within_bbox,
                    target_in_bbox_flag,
                    original_image_description,
                    bbox_within_the_cropped_image,
                    is_new_cropped,

                ]
            )
            assert all(attr is not None for attr in [parent, node_value, rephrased_objection_descriptions])
    except AssertionError:
        print(f"Instantiating node with type {node_type} failed!")
        breakpoint()
        exit()

def keep_track_of_paraphrasing(MLLM_Node):
    if MLLM_Node.node_type is Node_Type.USER_QUESTION:
        MLLM_Node.paraphrased = False
    elif MLLM_Node.node_type is Node_Type.REPHRASED_OBJECTION_DESCRIPTION:
        MLLM_Node.paraphrased = True

        MLLM_Node.object_descriptions = MLLM_Node.rephrased_object_descriptions
    else:
        assert MLLM_Node.parent is not None
        MLLM_Node.paraphrased = MLLM_Node.parent.paraphrased

def record_number_of_times_images_cropped_with_bbox(MLLM_Node):
    if MLLM_Node.parent is None:
        MLLM_Node.cropped_image_counter = 0
    else:
        if MLLM_Node.node_type is Node_Type.LOCATION_WITHIN_BBOX and MLLM_Node.is_new_cropped:
            MLLM_Node.cropped_image_counter = MLLM_Node.parent.cropped_image_counter + 1
        else:
            MLLM_Node.cropped_image_counter = MLLM_Node.parent.cropped_image_counter

def record_solution_trace_from_root_to_now(MLLM_Node):
    if MLLM_Node.parent is None:
        assert MLLM_Node.node_type is Node_Type.USER_QUESTION
        MLLM_Node.solution_trace = {0: {"user_question": MLLM_Node.user_question, "question_image": MLLM_Node.image_path, "object_descriptions": MLLM_Node.object_descriptions}}
    else:
        assert MLLM_Node.node_type is not Node_Type.USER_QUESTION
        MLLM_Node.solution_trace = deepcopy(MLLM_Node.parent.solution_trace)

        if MLLM_Node.node_type is Node_Type.DIRECT_LOCATION:
            assert MLLM_Node.cropped_image_counter in MLLM_Node.solution_trace.keys()
            assert MLLM_Node.cropped_image_counter == MLLM_Node.parent.cropped_image_counter
            MLLM_Node.solution_trace[MLLM_Node.cropped_image_counter]["direct_location"] = {
                "bbox": MLLM_Node.direct_bbox,
                "value": MLLM_Node.node_value,
            }
        elif MLLM_Node.node_type is Node_Type.BBOX_DESCRIPTION:
            assert MLLM_Node.cropped_image_counter in MLLM_Node.solution_trace.keys()
            assert MLLM_Node.cropped_image_counter == MLLM_Node.parent.cropped_image_counter
            MLLM_Node.solution_trace[MLLM_Node.cropped_image_counter]["descriptions_within_bbox"] = {
                "descriptions": MLLM_Node.descriptions_within_bbox,
                "value": MLLM_Node.node_value,
            }
        elif MLLM_Node.node_type is Node_Type.ORIGINAL_IMAGE_DESCRIPTION:
            assert MLLM_Node.parent.node_type is Node_Type.USER_QUESTION, "A3 (ORIGINAL_IMAGE_DESCRIPTION) can only serve as the initial node following the root node"
            assert MLLM_Node.cropped_image_counter in MLLM_Node.solution_trace.keys()
            assert MLLM_Node.cropped_image_counter == 0
            MLLM_Node.solution_trace[MLLM_Node.cropped_image_counter]["original_image_description"] = {
                "descriptions": MLLM_Node.original_image_description,
                "value": MLLM_Node.node_value,
            }
        elif MLLM_Node.node_type is Node_Type.LOCATION_WITHIN_BBOX:
            if MLLM_Node.cropped_image_counter not in MLLM_Node.solution_trace:
                MLLM_Node.solution_trace[MLLM_Node.cropped_image_counter] = {}
            MLLM_Node.solution_trace[MLLM_Node.cropped_image_counter]["bbox_within_the_cropped_image"] = {
                "bbox": MLLM_Node.bbox_within_the_cropped_image,
                "value": MLLM_Node.node_value,
                "is_new_cropped": MLLM_Node.is_new_cropped,
            }
        elif MLLM_Node.node_type is Node_Type.REPHRASED_OBJECTION_DESCRIPTION:
            assert MLLM_Node.cropped_image_counter in MLLM_Node.solution_trace.keys()
            assert MLLM_Node.cropped_image_counter == MLLM_Node.parent.cropped_image_counter
            MLLM_Node.solution_trace[MLLM_Node.cropped_image_counter]["rephrased_objection_descriptions"] = {
                "descriptions": MLLM_Node.rephrased_object_descriptions,
                "value": MLLM_Node.node_value,
            }

def potential_score_for_intermediate_nodes(MLLM_Node, potential_answers):
    if MLLM_Node.enable_potential_score:
        MLLM_Node.potential_answers = potential_answers
        MLLM_Node.potential_score = 0
        if MLLM_Node.parent is None:
            assert MLLM_Node.node_type is Node_Type.USER_QUESTION
            MLLM_Node.potential_answers_history = {}
        else:
            assert MLLM_Node.node_type is not Node_Type.USER_QUESTION
            MLLM_Node.potential_answers_history = deepcopy(MLLM_Node.parent.potential_answers_history)
            MLLM_Node.potential_answers_history[MLLM_Node.depth] = potential_answers
