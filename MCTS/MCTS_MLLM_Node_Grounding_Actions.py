from run_src.mstar_Grounding_utils import (
    Node_Type,

)
from copy import deepcopy


def mcts_Grounding_action_rules(MLLM_Node):
    action_type_dict = {
        "A1_DA": 'generate_direct_locations',
        'A2_BD': 'generate_bbox_descriptions',
        'A3_OID': 'generate_original_image_descriptions',
        'A4_LB': 'generate_locations_within_bbox',
        'A5_ROD': 'generate_rephrased_objection_descriptions',
    }

    assert MLLM_Node.node_type in [Node_Type.USER_QUESTION, Node_Type.DIRECT_LOCATION, Node_Type.BBOX_DESCRIPTION, Node_Type.ORIGINAL_IMAGE_DESCRIPTION, Node_Type.LOCATION_WITHIN_BBOX, Node_Type.REPHRASED_OBJECTION_DESCRIPTION], "An unsupported node type was used"

    action_list = []
    if MLLM_Node.node_type == Node_Type.USER_QUESTION:
        action_list.extend([
            action_type_dict['A1_DA'],
            action_type_dict["A3_OID"],
            action_type_dict["A5_ROD"]
        ])

    elif MLLM_Node.node_type == Node_Type.DIRECT_LOCATION:
        action_list.extend([
            action_type_dict["A2_BD"],
            action_type_dict["A4_LB"],
            action_type_dict["A5_ROD"]
        ])

    elif MLLM_Node.node_type == Node_Type.BBOX_DESCRIPTION:
        if MLLM_Node.target_in_bbox_flag:
            action_list.extend([
                action_type_dict["A4_LB"],
            ])
        else:
            action_list.extend([
                action_type_dict["A1_DA"],
                action_type_dict["A5_ROD"]
            ])

    elif MLLM_Node.node_type == Node_Type.ORIGINAL_IMAGE_DESCRIPTION:
        if MLLM_Node.target_existence_flag:
            action_list.extend([
                action_type_dict["A1_DA"],
            ])
        else:
            action_list.extend([
                action_type_dict["A5_ROD"],
            ])

    elif MLLM_Node.node_type == Node_Type.LOCATION_WITHIN_BBOX:
        action_list.extend([
            action_type_dict["A2_BD"],
            action_type_dict["A4_LB"],
        ])

    elif MLLM_Node.node_type == Node_Type.REPHRASED_OBJECTION_DESCRIPTION:
        action_list.append(action_type_dict["A1_DA"])

    assert len(action_list) > 0, "No permitted child node types available"

    return action_list

def auto_do_Grounding_action(MLLM_Node, action_type):
    action_data_list = []

    if action_type == "generate_direct_locations":
        (direct_answer_list, record_bbox_list, value_list, wrong_terminal_node_flag_list, is_new_cropped_list) = MLLM_Node.generator.generate_direct_location(
            user_question=MLLM_Node.user_question, object_descriptions = MLLM_Node.object_descriptions, paraphrased=MLLM_Node.paraphrased, question_image=MLLM_Node.question_image, depth = MLLM_Node.depth+1, bbox_type=MLLM_Node.bbox_type
        )
        assert len(direct_answer_list) == len(record_bbox_list) == len(value_list) == len(wrong_terminal_node_flag_list), "The lengths of generated answers and values are inconsistent"
        for i in range(len(direct_answer_list)):
            action_data = {
                "parent": MLLM_Node,
                "depth": MLLM_Node.depth + 1,
                "node_type": Node_Type.DIRECT_LOCATION,
                "node_value": value_list[i],
                "direct_bbox": direct_answer_list[i],
                "record_bbox": record_bbox_list[i],
                "wrong_terminal_node_flag": wrong_terminal_node_flag_list[i],
                "is_new_cropped": is_new_cropped_list[i],
            }
            action_data_list.append(action_data)

    elif action_type == "generate_bbox_descriptions":
        (descriptions_within_bbox_list, target_in_bbox_flag_list, value_list) = MLLM_Node.generator.generate_bbox_description(
            user_question=MLLM_Node.user_question, object_descriptions=MLLM_Node.object_descriptions, question_image=MLLM_Node.question_image, record_bbox = MLLM_Node.record_bbox
        )
        assert len(descriptions_within_bbox_list) == len(target_in_bbox_flag_list) == len(value_list), "The lengths of generated sub-questions and sub-answers are inconsistent"
        for i in range(len(descriptions_within_bbox_list)):
            action_data = {
                "parent": MLLM_Node,
                "depth": MLLM_Node.depth + 1,
                "node_type": Node_Type.BBOX_DESCRIPTION,
                "node_value": value_list[i],
                "descriptions_within_bbox": descriptions_within_bbox_list[i],
                "target_in_bbox_flag": target_in_bbox_flag_list[i],
            }
            action_data_list.append(action_data)

    elif action_type == "generate_original_image_descriptions":
        (original_image_description_list, target_existence_flag_list, value_list) = MLLM_Node.generator.generate_original_image_description(
            user_question=MLLM_Node.user_question,
            object_descriptions=MLLM_Node.object_descriptions,
            question_image=MLLM_Node.question_image
        )
        assert len(original_image_description_list) == len(value_list) == len(target_existence_flag_list), "The lengths of generated sub-questions and sub-answers are inconsistent"
        for i in range(len(original_image_description_list)):
            action_data = {
                "parent": MLLM_Node,
                "depth": MLLM_Node.depth + 1,
                "node_type": Node_Type.ORIGINAL_IMAGE_DESCRIPTION,
                "node_value": value_list[i],
                "original_image_description": original_image_description_list[i],
                "target_existence_flag": target_existence_flag_list[i],
            }
            action_data_list.append(action_data)
    elif action_type == "generate_locations_within_bbox":
        bbox_within_bbox_list, is_new_cropped_list, value_list, wrong_terminal_node_flag_list, record_bbox_list = MLLM_Node.generator.generate_location_within_bbox(
            user_question=MLLM_Node.user_question,
            object_descriptions=MLLM_Node.object_descriptions,
            question_image=MLLM_Node.question_image,
            record_bbox=MLLM_Node.record_bbox,
            depth=MLLM_Node.depth + 1,
            bbox_type=MLLM_Node.bbox_type
        )
        assert len(bbox_within_bbox_list) == len(is_new_cropped_list) == len(value_list) == len(wrong_terminal_node_flag_list) == len(record_bbox_list), "The lengths of generated sub-questions and sub-answers are inconsistent"
        for i in range(len(bbox_within_bbox_list)):
            action_data = {
                "parent": MLLM_Node,
                "depth": MLLM_Node.depth + 1,
                "node_type": Node_Type.LOCATION_WITHIN_BBOX,
                "bbox_within_the_cropped_image": bbox_within_bbox_list[i],
                "is_new_cropped": is_new_cropped_list[i],
                "node_value": value_list[i],
                "record_bbox": record_bbox_list[i],
                "wrong_terminal_node_flag": wrong_terminal_node_flag_list[i],
            }
            action_data_list.append(action_data)
    elif action_type == "generate_rephrased_objection_descriptions":
        rephrased_object_descriptions_list, value_list, wrong_terminal_node_flag_list = MLLM_Node.generator.generate_rephrased_objection_description(
            user_question=MLLM_Node.user_question,
            object_descriptions=MLLM_Node.object_descriptions,
            question_image=MLLM_Node.question_image,
            paraphrased_flag=MLLM_Node.paraphrased,
            raw_object_descriptions=MLLM_Node.raw_object_descriptions,
        )
        assert len(rephrased_object_descriptions_list) == len(value_list), "The lengths of generated sub-questions and sub-answers are inconsistent"

        for i in range(len(rephrased_object_descriptions_list)):
            action_data = {
                "parent": MLLM_Node,
                "depth": MLLM_Node.depth + 1,
                "node_type": Node_Type.REPHRASED_OBJECTION_DESCRIPTION,
                "node_value": value_list[i],
                "rephrased_object_descriptions": deepcopy(rephrased_object_descriptions_list[i]),
                "wrong_terminal_node_flag": wrong_terminal_node_flag_list[i],
            }
            action_data_list.append(action_data)
    elif MLLM_Node.node_type == Node_Type.DIRECT_ANSWER or action_type == "there_is_no_child_node":
        raise ValueError("DIRECT_ANSWER node cannot create children!!")

    return action_data_list
