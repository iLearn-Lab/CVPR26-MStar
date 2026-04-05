import sys

sys.path.append(".")

from tqdm import trange
from typing import List, Dict, Tuple
from PIL import Image
import torch
import copy

try:
    from rapidfuzz import fuzz, process
except:
    pass

from MLLM_API.IO_System import IO_System
from common.utils import read_txt, read_json
from eval_src.Evaluator import Evaluator
from MCTS.MCTS_backbone import MCTS_Searcher, MCTS_Node


from run_src.mstar_Grounding_utils import (
    Node_Type,
    stochastic_find_Grounding_solution,
    extract_solution_from_node,
    convert_bbox_to_absolute
)

from MCTS.MCTS_MLLM_Grounding_tools import (
    node_initial_state_check,
    keep_track_of_paraphrasing,
    record_number_of_times_images_cropped_with_bbox,

    record_solution_trace_from_root_to_now,
    potential_score_for_intermediate_nodes,
)

from MCTS.MCTS_MLLM_Node_Grounding_Actions import (
    mcts_Grounding_action_rules,
    auto_do_Grounding_action,

)

def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)

class MLLM_Grounding_Generator:
    def __init__(self, args, tokenizer, model, evaluator: Evaluator) -> None:
        self.io = IO_System(args, tokenizer, model)
        self.evaluator = evaluator

        self.num_a1_steps = args.num_a1_steps
        self.num_a2_steps = args.num_a2_steps
        self.num_a3_steps = args.num_a3_steps
        self.num_a4_steps = args.num_a4_steps
        self.num_a5_steps = args.num_a5_steps

        self.max_depth_allowed = args.max_depth_allowed
        self.max_tokens = args.max_tokens
        self.enable_potential_score = args.enable_potential_score
        self.mcts_num_last_votes = args.mcts_num_last_votes

        self.args = args

        self.direct_location_prompt = read_txt(args.direct_location_prompt_path)
        self.bbox_description_prompt = read_txt(args.bbox_description_prompt_path)
        self.original_image_description_prompt = read_txt(args.original_image_description_prompt_path)
        self.location_within_bbox_prompt = read_txt(args.location_within_bbox_prompt_path)
        self.rephrased_objection_description_prompt = read_txt(args.rephrased_objection_description_prompt_path)

        if args.api == "transformers":
            self.transformers_prompt_template = read_json(args.transformers_prompt_template_path)
        else:
            raise ValueError("Invalid API type")

    def _generate_input_prompt_list(self, prompt_template: dict, question: str, question_image, system_prompt=None):
        if self.args.api == "transformers":
            assert len(prompt_template) <= 2, "transformers prompt template should be less than 2"
            io_input = prompt_template["prompt_template"]
            for role_prompt in io_input:
                if role_prompt['role'] == 'system':
                    for type_prompt in role_prompt['content']:
                        if type_prompt['type'] == "text":
                            type_prompt['text'] = system_prompt
                elif role_prompt['role'] == 'user':
                    for type_prompt in role_prompt['content']:
                        if type_prompt['type'] == 'text':
                            type_prompt['text'] = question
                        elif type_prompt['type'] == 'image':
                            type_prompt['image'] = question_image
        else:
            raise ValueError("Invalid role type in prompt template")

        return io_input

    def _generator_is_terminal(self, depth, max_depth_allowed):
        return depth >= max_depth_allowed

    def _ensure_min_bbox_size(self, extend_bbox, question_image, min_size=28):
        extend_bbox = [int(round(x)) for x in extend_bbox]

        img_width, img_height = question_image.size

        current_width = extend_bbox[2] - extend_bbox[0]
        current_height = extend_bbox[3] - extend_bbox[1]

        if current_width < min_size:
            width_expansion = (min_size - current_width) // 2
            width_remainder = (min_size - current_width) % 2

            new_x1 = max(0, extend_bbox[0] - width_expansion - width_remainder)

            new_x2 = min(img_width, extend_bbox[2] + width_expansion)

            if new_x2 == img_width and (new_x2 - new_x1) < min_size:
                new_x1 = max(0, new_x2 - min_size)

            elif new_x1 == 0 and (new_x2 - new_x1) < min_size:
                new_x2 = min(img_width, new_x1 + min_size)

            extend_bbox[0] = new_x1
            extend_bbox[2] = new_x2

        if current_height < min_size:
            height_expansion = (min_size - current_height) // 2
            height_remainder = (min_size - current_height) % 2

            new_y1 = max(0, extend_bbox[1] - height_expansion - height_remainder)

            new_y2 = min(img_height, extend_bbox[3] + height_expansion)

            if new_y2 == img_height and (new_y2 - new_y1) < min_size:
                new_y1 = max(0, new_y2 - min_size)

            elif new_y1 == 0 and (new_y2 - new_y1) < min_size:
                new_y2 = min(img_height, new_y1 + min_size)

            extend_bbox[1] = new_y1
            extend_bbox[3] = new_y2

        return extend_bbox

    def generate_direct_location(self, user_question, object_descriptions, paraphrased, question_image, depth, bbox_type):
        direct_bbox_list = []
        node_value_list = []
        wrong_terminal_node_flag_list = []
        is_new_cropped_list = []

        if self._generator_is_terminal(depth, self.max_depth_allowed):
            num_return = self.mcts_num_last_votes
        else:
            num_return = self.num_a1_steps

        question_prompt = "Please locate the region this sentence describes in the image: " + object_descriptions[0]

        system_prompt = self.direct_location_prompt

        io_input = self._generate_input_prompt_list(
            self.transformers_prompt_template,
            question_prompt,
            question_image,
            system_prompt
        )

        io_output_list = self.io.generate(
            io_input,
            num_return=num_return,
            max_tokens=self.max_tokens,
        )

        if self.args.debug_print_flag:
            print("Direct Location IO Output:")
            for i, x in enumerate(io_output_list):
                print(f"io_output{i}:", x)

        bbox_label_list = self.evaluator._extract_json_blocks(io_output_list)

        bbox_tensor = []
        for bbox_label in bbox_label_list:
            bbox = bbox_label["bbox_2d"]
            bbox = convert_bbox_to_absolute(bbox, bbox_type, image_width=question_image.size[0], image_height=question_image.size[1])
            bbox_tensor.append(bbox)
        bbox_tensor = torch.tensor(bbox_tensor, dtype=torch.float32)

        merged_boxes, counts = self.evaluator._merge_bboxes_with_count(bbox_tensor, method="iou_cluster", iou_threshold=self.args.merge_bboxes_iou_threshold)

        max_index = max(enumerate(counts), key=lambda x: x[1])[0]
        direct_bbox = [int(round(x.item())) for x in merged_boxes[max_index]]

        direct_bbox, wrong_terminal_node_flag = self.evaluator._inspect_and_clip_bbox_with_iou(direct_bbox, question_image.size, iou_threshold=self.args.bbox2image_iou_threshold)

        if not wrong_terminal_node_flag:
            node_value = counts[max_index] / len(bbox_label_list)

            is_new_cropped = True
            is_new_cropped_list.append(is_new_cropped)
        else:
            node_value = float("-inf")

            is_new_cropped = False
            is_new_cropped_list.append(is_new_cropped)

        direct_bbox_list.append(direct_bbox)
        node_value_list.append(node_value)
        record_bbox_list = copy.deepcopy(direct_bbox_list)
        wrong_terminal_node_flag_list.append(wrong_terminal_node_flag)

        return direct_bbox_list, record_bbox_list, node_value_list, wrong_terminal_node_flag_list, is_new_cropped_list

    def generate_bbox_description(self, user_question, object_descriptions, question_image, record_bbox):
        descriptions_within_bbox_list = []
        target_in_bbox_flag_list = []
        node_value_list = []

        num_return = self.num_a2_steps

        original_descriptions_str = "[" + ", ".join([f"'{desc}'" for desc in object_descriptions]) + "]"

        assert record_bbox is not None, "record_bbox should not be None"

        extend_bbox = [
            int(record_bbox[0] - (record_bbox[2] - record_bbox[0]) * self.args.bbox_extend_ratio),
            int(record_bbox[1] - (record_bbox[3] - record_bbox[1]) * self.args.bbox_extend_ratio),
            int(record_bbox[2] + (record_bbox[2] - record_bbox[0]) * self.args.bbox_extend_ratio),
            int(record_bbox[3] + (record_bbox[3] - record_bbox[1]) * self.args.bbox_extend_ratio)
        ]

        extend_bbox[0] = max(0, extend_bbox[0])
        extend_bbox[1] = max(0, extend_bbox[1])
        extend_bbox[2] = min(question_image.size[0], extend_bbox[2])
        extend_bbox[3] = min(question_image.size[1], extend_bbox[3])

        extend_bbox = [int(round(x)) for x in extend_bbox]
        extend_bbox = self._ensure_min_bbox_size(extend_bbox, question_image, self.args.bbox_extend_min_size)

        bbox_image = question_image.crop(extend_bbox)

        question_prompt = "First, please describe the instance objects in the image using a Markdown code block format. Then, carefully interpret the descriptions " + original_descriptions_str + ", which refer to a specific instance object. If that instance appears in the image, answer True; otherwise, answer False."
        system_prompt = self.bbox_description_prompt

        io_input = self._generate_input_prompt_list(
            self.transformers_prompt_template,
            question_prompt,
            bbox_image,
            system_prompt
        )
        io_output_list = self.io.generate(
            io_input,
            num_return=num_return,
            max_tokens=self.max_tokens,
        )

        if self.args.debug_print_flag:
            print("BBox Description Output:")
            for i, x in enumerate(io_output_list):
                print(f"io_output{i}:", x)

        objects_within_image_sublist = []
        key_of_original_descriptions_sublist = []
        target_in_bbox_flag_sublist = []
        for md_str in io_output_list:
            code_blocks_list = self.evaluator._extract_code_blocks_from_markdown(md_str)
            if len(code_blocks_list) < 2:
                continue
            if code_blocks_list[-1] not in ["True", "False"]:
                continue
            objects_within_image = str(code_blocks_list[0]).strip().split("\n")

            target_in_bbox_flag = self.evaluator._str_to_bool(str(code_blocks_list[-1]))
            objects_within_image_sublist.append(objects_within_image)

            target_in_bbox_flag_sublist.append(target_in_bbox_flag)

        for objects_list in objects_within_image_sublist:
            if len(objects_list) == 0:
                continue
            for object_str in objects_list:
                if object_str == "":
                    continue

                instance_str = None
                instance_descriptions = self.evaluator._extract_bold_text_simple(object_str)
                if instance_descriptions:
                    instance_str = instance_descriptions[0]

                if instance_str is not None and instance_str not in descriptions_within_bbox_list:
                    descriptions_within_bbox_list.append(instance_str)
        descriptions_within_bbox_list = [list(set(descriptions_within_bbox_list))]

        true_count = target_in_bbox_flag_sublist.count(True)
        node_value = (true_count / len(target_in_bbox_flag_sublist)) if len(target_in_bbox_flag_sublist) > 0 else 0
        node_value_list.append(node_value)

        if node_value > self.args.obj_within_bbox_num_threshold:
            target_in_bbox_flag = True
        else:
            target_in_bbox_flag = False
        target_in_bbox_flag_list.append(target_in_bbox_flag)

        return descriptions_within_bbox_list, target_in_bbox_flag_list, node_value_list

    def generate_original_image_description(self, user_question, object_descriptions, question_image):
        original_image_description_list = []
        target_existence_flag_list = []
        node_value_list = []

        num_return = self.num_a3_steps

        original_descriptions_str = "[" + ", ".join([f"'{desc}'" for desc in object_descriptions]) + "]"

        question_prompt = "First, please describe the instance objects in the image. Second, carefully interpret the descriptions " + original_descriptions_str + ", which refer to a specific instance object. If that instance appears in the image, answer True; otherwise, answer False."
        system_prompt  = self.original_image_description_prompt

        io_input = self._generate_input_prompt_list(
            self.transformers_prompt_template,
            question_prompt,
            question_image,
            system_prompt
        )

        io_output_list = self.io.generate(
            io_input,
            num_return=num_return,
            max_tokens=self.max_tokens,
        )

        if self.args.debug_print_flag:
            print("Original Image Description Output:")
            for i, x in enumerate(io_output_list):
                print(f"io_output{i}:", x)

        original_image_description_sublist = []
        existence_check_results_list = []

        for md_str in io_output_list:
            code_blocks_list = self.evaluator._extract_code_blocks_from_markdown(md_str)
            if len(code_blocks_list) < 2:
                continue
            if code_blocks_list[-1] not in ["True", "False"]:
                continue
            original_image_description_sublist.append(code_blocks_list[0].strip())
            target_existence_flag = self.evaluator._str_to_bool(code_blocks_list[-1])
            existence_check_results_list.append(target_existence_flag)

        true_count = existence_check_results_list.count(True)
        node_value = true_count / len(existence_check_results_list) if len(existence_check_results_list) > 0 else 0
        node_value_list.append(node_value)

        if node_value >= self.args.obj_within_image_num_threshold:
            target_existence_flag = True
        else:
            target_existence_flag = False
        target_existence_flag_list.append(target_existence_flag)

        target_existence_indices = [i for i, flag in enumerate(existence_check_results_list) if flag == target_existence_flag]
        if target_existence_indices:
            descriptions_with_target = [original_image_description_sublist[i] for i in target_existence_indices]

            longest_description = max(descriptions_with_target, key=len)
            original_image_description_list.append(longest_description)
        else:
            original_image_description_list.append("")

        return original_image_description_list, target_existence_flag_list, node_value_list

    def generate_location_within_bbox(self, user_question, object_descriptions, question_image, record_bbox, depth, bbox_type):
        bbox_within_bbox_list = []
        wrong_terminal_node_flag_list = []
        is_new_cropped_list = []
        node_value_list = []
        record_bbox_list = []

        if self._generator_is_terminal(depth, self.max_depth_allowed):
            num_return = self.mcts_num_last_votes
        else:
            num_return = self.num_a4_steps

        assert record_bbox is not None, "record_bbox should not be None"

        extend_bbox = [
            int(record_bbox[0] - (record_bbox[2] - record_bbox[0]) * self.args.bbox_extend_ratio),
            int(record_bbox[1] - (record_bbox[3] - record_bbox[1]) * self.args.bbox_extend_ratio),
            int(record_bbox[2] + (record_bbox[2] - record_bbox[0]) * self.args.bbox_extend_ratio),
            int(record_bbox[3] + (record_bbox[3] - record_bbox[1]) * self.args.bbox_extend_ratio)
        ]

        extend_bbox[0] = max(0, extend_bbox[0])
        extend_bbox[1] = max(0, extend_bbox[1])
        extend_bbox[2] = min(question_image.size[0], extend_bbox[2])
        extend_bbox[3] = min(question_image.size[1], extend_bbox[3])

        extend_bbox = [int(round(x)) for x in extend_bbox]
        extend_bbox = self._ensure_min_bbox_size(extend_bbox, question_image, self.args.bbox_extend_min_size)
        bbox_image = question_image.crop(extend_bbox)

        original_descriptions_str = "[" + ", ".join([f"'{desc}'" for desc in object_descriptions]) + "]"

        question_prompt = "Please locate the region this sentence describes in the image: " + object_descriptions[0]

        system_prompt = self.direct_location_prompt

        io_input = self._generate_input_prompt_list(
            self.transformers_prompt_template,
            question_prompt,
            bbox_image,
            system_prompt
        )

        io_output_list = self.io.generate(
            io_input,
            num_return=num_return,
            max_tokens=self.max_tokens,
        )

        if self.args.debug_print_flag:
            print("Location Within BBox Generation Output:")
            for i, x in enumerate(io_output_list):
                print(f"io_output{i}:", x)

        bbox_label_list = self.evaluator._extract_json_blocks(io_output_list)

        bbox_tensor = []
        for bbox_label in bbox_label_list:
            bbox = bbox_label["bbox_2d"]
            bbox = convert_bbox_to_absolute(bbox, bbox_type, image_width=question_image.size[0], image_height=question_image.size[1])
            bbox_tensor.append(bbox)
        bbox_tensor = torch.tensor(bbox_tensor, dtype=torch.float32)

        merged_boxes, counts = self.evaluator._merge_bboxes_with_count(bbox_tensor, method="iou_cluster", iou_threshold=self.args.merge_bboxes_iou_threshold)

        max_index = max(enumerate(counts), key=lambda x: x[1])[0]
        bbox_within_bbox = [int(round(x.item())) for x in merged_boxes[max_index]]

        bbox_within_bbox, wrong_terminal_node_flag = self.evaluator._inspect_and_clip_bbox_with_iou(
            bbox_within_bbox,
            bbox_image.size,
            iou_threshold=self.args.bbox2image_iou_threshold
        )

        bbox_within_bbox = [
            bbox_within_bbox[0] + extend_bbox[0],
            bbox_within_bbox[1] + extend_bbox[1],
            bbox_within_bbox[2] + extend_bbox[0],
            bbox_within_bbox[3] + extend_bbox[1]
        ]

        if not wrong_terminal_node_flag:
            node_value = counts[max_index] / len(bbox_label_list)

            bbox_within_bbox_list.append(bbox_within_bbox)

            record_bbox_list.append(bbox_within_bbox)

            is_new_cropped = True
            is_new_cropped_list.append(is_new_cropped)
        else:
            node_value = float("-inf")

            bbox_within_bbox_list.append(record_bbox)

            record_bbox_list.append(record_bbox)

            is_new_cropped = False
            is_new_cropped_list.append(is_new_cropped)
        node_value_list.append(node_value)
        wrong_terminal_node_flag_list.append(wrong_terminal_node_flag)

        return bbox_within_bbox_list, is_new_cropped_list, node_value_list, wrong_terminal_node_flag_list, record_bbox_list

    def generate_rephrased_objection_description(self, user_question, object_descriptions, question_image, paraphrased_flag, raw_object_descriptions):
        rephrased_object_descriptions_list = []
        rephrased_user_question_list = []
        node_value_list = []
        wrong_terminal_node_flag_list = []

        if paraphrased_flag:
            object_descriptions = object_descriptions + raw_object_descriptions
        else:
            object_descriptions = object_descriptions
        num_return = self.num_a5_steps
        original_descriptions_str = "[" + ", ".join([f"'{desc}'" for desc in object_descriptions]) + "]"

        question_prompt = "First, please describe the instance objects in the image using a Markdown code block format. Second, based on the previous descriptions for the specified target instance " + original_descriptions_str + ", identify the instance object in the image that best matches them, and provide an improved description according to the image content."
        system_prompt = self.rephrased_objection_description_prompt

        io_input = self._generate_input_prompt_list(
            self.transformers_prompt_template,
            question_prompt,
            question_image,
            system_prompt
        )
        io_output_list = self.io.generate(
            io_input,
            num_return=num_return,
            max_tokens=self.max_tokens,
        )

        if self.args.debug_print_flag:
            print("Rephrased Object Descriptions Output:")
            for i, x in enumerate(io_output_list):
                print(f"io_output{i}:", x)

        rephrased_object_descriptions_sublist = []
        for md_str in io_output_list:
            code_blocks_list = self.evaluator._extract_code_blocks_from_markdown(md_str)
            if len(code_blocks_list) < 2:
                continue
            rephrased_object_descriptions_sublist.append(code_blocks_list[1].strip())

        if len(rephrased_object_descriptions_sublist) > 0:
            try:
                most_similar_texts, avg_score = self.evaluator._get_similar_texts(
                    rephrased_object_descriptions_sublist,
                    top_n=self.args.rephrased_description_top_n
                )
                node_value = avg_score
                wrong_terminal_node_flag = False

            except ValueError as e:
                print(f"Error in generating rephrased object descriptions: {e}")
                most_similar_texts = raw_object_descriptions
                node_value = float("-inf")
                wrong_terminal_node_flag = True

        rephrased_object_descriptions_list.append(most_similar_texts)
        node_value_list.append(node_value)
        wrong_terminal_node_flag_list.append(wrong_terminal_node_flag)

        return rephrased_object_descriptions_list, node_value_list, wrong_terminal_node_flag_list

class MLLM_Grounding_MCTS_Node(MCTS_Node):
    def __init__(
        self,
        parent: "MLLM_Grounding_MCTS_Node",
        depth: int,
        node_type: Node_Type,
        verbose: bool = False,

        node_value: float = None,
        generator: MLLM_Grounding_Generator = None,
        user_question: str = None,
        raw_object_descriptions: list = None,
        object_descriptions: list = None,
        question_image = None,
        image_path = None,
        max_depth_allowed: int = None,
        wrong_terminal_node_flag: bool = False,
        expected_answer: list  = None,

        direct_bbox: list = None,

        record_bbox: list = None,

        descriptions_within_bbox: list = None,
        target_in_bbox_flag: bool = None,

        original_image_description: str = None,
        target_existence_flag: bool = None,

        bbox_within_the_cropped_image: list = None,
        is_new_cropped: bool = None,

        rephrased_object_descriptions: list = None,

        enable_potential_score: bool = None,
        potential_answers: List[str] = None,
        bbox_type: str = "absolute",

    ) -> None:
        super().__init__()

        assert node_type in [
            Node_Type.USER_QUESTION,
            Node_Type.DIRECT_LOCATION,
            Node_Type.BBOX_DESCRIPTION,
            Node_Type.ORIGINAL_IMAGE_DESCRIPTION,
            Node_Type.LOCATION_WITHIN_BBOX,
            Node_Type.REPHRASED_OBJECTION_DESCRIPTION,
            ], f"{node_type} is not a valid Node type for the Grounding task; a Grounding-compatible node type must be used"

        node_initial_state_check(parent, depth, node_type, node_value, generator, user_question, raw_object_descriptions, object_descriptions, question_image, max_depth_allowed, wrong_terminal_node_flag, expected_answer, direct_bbox, record_bbox, descriptions_within_bbox, target_in_bbox_flag, original_image_description, target_existence_flag, bbox_within_the_cropped_image, is_new_cropped, rephrased_object_descriptions)

        self.parent = parent
        self.children: List["MLLM_Grounding_MCTS_Node"] = []
        self.depth = depth
        self.node_type = node_type
        self.node_value = node_value

        self.direct_bbox = direct_bbox

        self.descriptions_within_bbox = descriptions_within_bbox
        self.target_in_bbox_flag = target_in_bbox_flag
        self.original_image_description = original_image_description
        self.target_existence_flag = target_existence_flag
        self.bbox_within_the_cropped_image = bbox_within_the_cropped_image
        self.is_new_cropped = is_new_cropped

        self.rephrased_object_descriptions = rephrased_object_descriptions
        self.paraphrased = False
        self.wrong_terminal_node_flag = wrong_terminal_node_flag

        if self.node_type in [Node_Type.USER_QUESTION, Node_Type.DIRECT_LOCATION, Node_Type.LOCATION_WITHIN_BBOX]:
            self.record_bbox = record_bbox
        else:
            self.record_bbox = parent.record_bbox

        if parent is None:
            self.verbose = verbose
            self.user_question = user_question
            self.raw_object_descriptions = raw_object_descriptions
            self.object_descriptions = raw_object_descriptions
            self.expected_answer = expected_answer
            self.generator = generator

            self.max_depth_allowed = max_depth_allowed
            self.enable_potential_score = enable_potential_score
            self.question_image = question_image
            self.image_path = image_path
            self.bbox_type = bbox_type
        else:
            self.verbose = parent.verbose
            self.user_question = parent.user_question
            self.raw_object_descriptions = parent.raw_object_descriptions
            self.object_descriptions = parent.object_descriptions
            self.expected_answer = parent.expected_answer
            self.generator = parent.generator

            self.max_depth_allowed = parent.max_depth_allowed
            self.enable_potential_score = parent.enable_potential_score
            self.question_image = parent.question_image
            self.image_path = parent.image_path
            self.bbox_type = parent.bbox_type

        keep_track_of_paraphrasing(self)

        record_number_of_times_images_cropped_with_bbox(self)

        record_solution_trace_from_root_to_now(self)

        potential_score_for_intermediate_nodes(self, potential_answers)

        self.action_list = mcts_Grounding_action_rules(self)

        pass

    def __str__(self) -> str:
        type2str = {
            Node_Type.USER_QUESTION: "UQ",
            Node_Type.DIRECT_LOCATION: "A1_DA",
            Node_Type.BBOX_DESCRIPTION: "A2_BD",
            Node_Type.ORIGINAL_IMAGE_DESCRIPTION: "A3_OID",
            Node_Type.LOCATION_WITHIN_BBOX: "A4_LB",
            Node_Type.REPHRASED_OBJECTION_DESCRIPTION: "A5_ROD",
        }
        return f"{type2str[self.node_type]}-{self.id}"

    def _create_children(self):
        for action_type in self.action_list:
            action_data_list = auto_do_Grounding_action(self, action_type)
            for action_data in action_data_list:
                self.children.append(
                        MLLM_Grounding_MCTS_Node(
                            **action_data,
                        )
                    )
        assert self.children
        return self.children

    def is_valid_leaf_node(self):
        if self.wrong_terminal_node_flag:
            return False
        elif not self.wrong_terminal_node_flag and self.node_type in [Node_Type.DIRECT_LOCATION, Node_Type.LOCATION_WITHIN_BBOX]:
            return True
        else:
            return False

    def is_wrong_terminal_node(self):
        if self.wrong_terminal_node_flag:
            return True
        elif not self.wrong_terminal_node_flag:
            return False

    def set_potential_score(self, score: float):
        self.potential_score = score

    def find_children(self, rollout_id: int):
        self.children = self.children or self._create_children()
        for child in self.children:
            child.set_rollout_id(rollout_id)
        assert self.children
        return self.children

    def is_terminal(self):
        return self.depth >= self.max_depth_allowed or self.is_wrong_terminal_node()

    def calculate_reward(self):
        if self.is_valid_leaf_node():
            assert self.node_value is not None, breakpoint()
            return self.node_value
        else:
            return 0

    def skip_backprop(self):
        return self.node_type is Node_Type.USER_QUESTION or self.node_type is Node_Type.REPHRASED_USER_QUESTION

def MLLM_search_for_Grounding(args, user_question: str, object_descriptions, question_id: int, gt_answer: str, generator: MLLM_Grounding_Generator, question_image, image_path):
    verbose_print(
        f"********************* Searching for answers to question {question_id} ********************* ", args.verbose
    )

    mcts_searcher = MCTS_Searcher(
        exploration_weight=args.mcts_exploration_weight,
        weight_scheduler=args.mcts_weight_scheduler,
        num_rollouts=args.num_rollouts,
        discount=args.mcts_discount_factor,
        verbose=args.verbose,
    )

    root_node = MLLM_Grounding_MCTS_Node(
        parent=None,
        depth=0,
        node_type=Node_Type.USER_QUESTION,
        verbose=args.verbose,
        generator=generator,
        user_question=user_question,
        question_image = question_image,
        raw_object_descriptions=object_descriptions,
        image_path=image_path,
        expected_answer=gt_answer,
        max_depth_allowed=args.max_depth_allowed,
        enable_potential_score=args.enable_potential_score,
        bbox_type=args.bbox_type,
    )

    model_rollout_merged_solutions = []
    model_rollout_all_solutions = []
    model_rollout_end_nodes = []
    model_rollout_end_solutions = []

    for i in (pbar := trange(args.num_rollouts, disable=True, position=0)):
        rollout_node = mcts_searcher.do_rollout(root_node, i)
        model_rollout_end_nodes.append(rollout_node)

        rollout_merged_solution, rollout_all_solutions, roolout_end_solution, rollout_all_solution_nodes, rollout_all_leaf_nodes = stochastic_find_Grounding_solution(
            root_node, rollout_node, generator.evaluator
        )
        model_rollout_merged_solutions.append(rollout_merged_solution)
        model_rollout_all_solutions.extend(rollout_all_solutions)
        if roolout_end_solution is not None:
            model_rollout_end_solutions.append(roolout_end_solution)

    seen = set()
    model_all_solutions_deduplicated = []

    for solution in model_rollout_all_solutions:
        key = (tuple(solution['bbox']), solution['score'])
        if key not in seen:
            seen.add(key)
            model_all_solutions_deduplicated.append(solution)

    return i, model_rollout_merged_solutions, model_all_solutions_deduplicated, model_rollout_end_solutions, rollout_all_solution_nodes, model_rollout_end_nodes, rollout_all_leaf_nodes

if __name__ == "__main__":
    image_path = "data/coco/train2014/COCO_train2014_000000527796.jpg"

    question_image = Image.open(image_path)

    user_question = "What is the color of the taxi cab?"

    object_descriptions = ["the taxi cab"]

    gt_answer = [[61, 410, 336, 528], [61, 410, 336, 52], [61, 410, 336, 5], [58, 412, 336, 552], [61, 410, 336, 560], [63, 409, 336, 532]]

    generator = MLLM_Grounding_Generator()
