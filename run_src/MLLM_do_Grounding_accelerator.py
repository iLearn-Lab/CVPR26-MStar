import sys
import os, json, time
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torchvision.ops as ops
import datetime
import copy
from common.utils import fix_seeds, setup_model_parallel, read_json
from common.arguments import get_parser, post_process_args, save_args
from common.email_send import send_error_email
from MCTS.MCTS_for_MLLM_Grounding import MLLM_Grounding_Generator, MLLM_search_for_Grounding
from accelerate import Accelerator
from accelerate.utils import gather_object, InitProcessGroupKwargs
from datasets import load_dataset
import traceback

sys.path.append(".")

def convert_data_format(source_dicts):
    converted_list = []
    for i, source_dict in enumerate(source_dicts):
        pil_image = source_dict.get('image')
        width, height = pil_image.size if pil_image else (0, 0)
        file_name = source_dict.get('file_name', '')

        try:
            image_id = int(file_name.split('_')[-2])
        except (ValueError, IndexError):
            image_id = -1

        base_image_info = {
            'coco_url': f'http://mscoco.org/images/{image_id}',
            'date_captured': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'file_name': file_name,
            'flickr_url': '',
            'height': height,
            'id': image_id,
            'license': -1,
            'width': width
        }

        bbox = source_dict.get('bbox', [0, 0, 0, 0])
        segmentation = source_dict.get('segmentation', [])
        if segmentation and not isinstance(segmentation[0], list):
            segmentation = [segmentation]

        base_anns = [{
            'area': bbox[2] * bbox[3] if len(bbox) == 4 else 0.0,
            'bbox': bbox,
            'category_id': -1,
            'id': int(source_dict.get('question_id', -1)),
            'image_id': image_id,
            'iscrowd': source_dict.get('iscrowd', 0),
            'segmentation': segmentation
        }]

        answers = source_dict.get('answer', [])
        for expand_id, raw_sentence in enumerate(answers):
            instruction = [{
                'raw': raw_sentence,
                'sent': raw_sentence.lower().strip('.'),
                'sent_id': expand_id,
                'tokens': raw_sentence.lower().strip('.').split()
            }]

            new_sample = {
                'image': file_name,
                'image_info': copy.deepcopy(base_image_info),
                'instruction': instruction,
                'anns': copy.deepcopy(base_anns),
                'new_img_id': i,
                'expand_id': expand_id,
                'image_pil': pil_image
            }
            converted_list.append(new_sample)

    return converted_list

def expand_object_descriptions(data_item_list):
    expanded_data_items = []
    for i, data_item in enumerate(data_item_list):
        instructions = data_item.get('instruction', [])
        if not instructions:
            continue

        original_id = i

        for idx, instruction in enumerate(instructions):
            new_data_item = copy.deepcopy(data_item)
            new_data_item['instruction'] = [instruction]
            new_data_item['expand_id'] = idx
            new_data_item['original_id'] = original_id
            expanded_data_items.append(new_data_item)

    return expanded_data_items

def preprocess_input_data(dataset, args):
    if args.expand_object_descriptions:
        proceed_data = expand_object_descriptions(dataset)

        print(f"Exploded dataset from {len(dataset)} to {len(proceed_data)} rows")
    else:
        proceed_data = dataset.map(lambda example, idx: {**example, "expand_id": idx}, with_indices=True)

    return proceed_data

def resize_image(image_pil, longest_length = 800):
    width, height = image_pil.size

    if width <= longest_length and height <= longest_length:
        return image_pil.copy()

    if width > height:
        new_width = longest_length
        new_height = int(height * (longest_length / width))
    else:
        new_height = longest_length
        new_width = int(width * (longest_length / height))

    resized_img = image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_img

def extract_Grounding_information(data_item, args, evaluator):
    if args.dataset_name in ['RefCOCO', 'RefCOCO+', 'RefCOCOg', "refcoco"] and args.text_file_type in ["parquet", "json"]:
        problem_id =data_item['anns'][0]['id']
        instruction_list = data_item['instruction']
        anns = data_item['anns'][0]
        sent_list = []
        if instruction_list is not None:
            for instruction in instruction_list:
                sent = instruction['sent']
                sent_list.append(sent)
            sent_summary = "[" + ", ".join([f"'{desc}'" for desc in sent_list]) + "]"
        else:
            raise ValueError("No question found in the data item.")
        object_descriptions = sent_list
        problem = "Please perform localization and output its bounding box coordinates using JSON format."
        gt_solution = data_item['anns'][0]
        gt_answer = data_item['anns'][0]['bbox']

        image_path = args.image_root + "/" + data_item['image_info']['file_name']
        image_pil = Image.open(image_path).convert("RGB")
        image_pil = resize_image(image_pil, longest_length=800)
        return problem_id, problem, object_descriptions, gt_solution, gt_answer, image_pil
    elif args.dataset_name in ['RefCOCO', 'RefCOCO+', 'RefCOCOg', "refcoco"] and args.text_file_type in ["parquet_floder"]:
        if args.expand_object_descriptions:
            problem_id = data_item['expand_id']
        else:
            problem_id =data_item['anns'][0]['id']
        instruction_list = data_item['instruction']
        anns = data_item['anns'][0]
        sent_list = []
        if instruction_list is not None:
            for instruction in instruction_list:
                sent = instruction['sent']
                sent_list.append(sent)
            sent_summary = "[" + ", ".join([f"'{desc}'" for desc in sent_list]) + "]"
        else:
            raise ValueError("No question found in the data item.")
        object_descriptions = sent_list
        problem = "Please perform localization and output its bounding box coordinates using JSON format."
        gt_solution = data_item['anns'][0]
        gt_answer = data_item['anns'][0]['bbox']
        image_pil = data_item['image_pil']
        image_pil = resize_image(image_pil, longest_length=800)
        return problem_id, problem, object_descriptions, gt_solution, gt_answer, image_pil
    else:
        problem_id, problem, gt_solution = data_item["id"], data_item["problem"], data_item["solution"]
        gt_answer = evaluator.extract_answer_from_gold_solution(gt_solution)
        return problem_id, problem, gt_solution, gt_answer, None

def apply_nms_coco(bboxes_list, scores_list, iou_threshold=0.5, score_threshold=0.1, sort_by_score=True):
    if len(bboxes_list) == 0:
        return [], []

    bboxes = torch.tensor(bboxes_list, dtype=torch.float32)
    scores = torch.tensor(scores_list, dtype=torch.float32)

    valid_indices = scores >= score_threshold
    bboxes = bboxes[valid_indices]
    scores = scores[valid_indices]

    if len(bboxes) == 0:
        return [], []

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 0] + bboxes[:, 2]
    y2 = bboxes[:, 1] + bboxes[:, 3]
    bboxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    keep_indices = ops.nms(bboxes_xyxy, scores, iou_threshold)

    filtered_bboxes_xyxy = bboxes_xyxy[keep_indices]
    filtered_scores = scores[keep_indices]

    filtered_bboxes = torch.zeros_like(filtered_bboxes_xyxy)
    filtered_bboxes[:, 0] = filtered_bboxes_xyxy[:, 0]
    filtered_bboxes[:, 1] = filtered_bboxes_xyxy[:, 1]
    filtered_bboxes[:, 2] = filtered_bboxes_xyxy[:, 2] - filtered_bboxes_xyxy[:, 0]
    filtered_bboxes[:, 3] = filtered_bboxes_xyxy[:, 3] - filtered_bboxes_xyxy[:, 1]

    if sort_by_score:
        sorted_indices = torch.argsort(filtered_scores, descending=True)
        filtered_bboxes = filtered_bboxes[sorted_indices]
        filtered_scores = filtered_scores[sorted_indices]

    return filtered_bboxes.tolist(), filtered_scores.tolist()

def main(args):
    fix_seeds(args.seed)

    kwargs_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=60000))
    accelerator = Accelerator(kwargs_handlers=[kwargs_handler])

    if args.model_parallel:
        args.local_rank, args.world_size = setup_model_parallel()
    else:
        args.local_rank, args.world_size = 0, 1

    if args.text_file_type == "json" and args.dataset_name not in ["RefCOCO", "refcoco"]:
        test_file = os.path.join(args.data_root, args.dataset_name, args.test_json_filename + ".json")
        assert os.path.exists(test_file), f"Test file {test_file} does not exist."
        data_item_list = read_json(test_file)
    elif args.text_file_type == "json" and args.dataset_name in ["RefCOCO", "refcoco"]:
        test_file = os.path.join(args.data_root, args.dataset_name, args.test_json_filename + ".json")
        data_item_list = load_dataset("json", data_files={args.test_json_filename: test_file})[args.test_json_filename]
    elif args.text_file_type == "parquet":
        test_file = os.path.join(args.data_root, args.dataset_name, args.test_json_filename+"*.parquet")
        data_item_list = load_dataset("parquet", data_files={args.test_json_filename: test_file})[args.test_json_filename]
    elif args.text_file_type == "parquet_floder":
        test_floder = os.path.join(args.data_root, args.dataset_name)
        data_item_list = load_dataset(test_floder)[args.test_json_filename]

    evaluator = eval(f"{args.evaluator_type}Evaluator()")

    tokenizer, model = None, None
    if args.api == "transformers":
        try:
            from transformers import set_seed
            set_seed(args.seed)
        except ImportError:
            print("The transformers library is not installed or is incompatible; unable to set random seed")

        from MLLM_API.transformers_API import load_transformers_model

        tokenizer, model = load_transformers_model(args.model_ckpt, device_map=accelerator.device, accelerator=accelerator)
    else:
        raise ValueError(f"Invalid API type: {args.api}")

    generator = MLLM_Grounding_Generator(args, tokenizer, model, evaluator)

    total_correct = 0
    total_correct_limit = 0
    num_tested = 0
    start_time = time.time()

    all_dataset = preprocess_input_data(data_item_list, args)
    if accelerator is not None:
        with accelerator.split_between_processes(all_dataset) as dataset:
            accelerator_question_json_list = []
            accelerator_data_item_list = []
            accelerator_final_coco_output_list = []
            accelerator_wrong_data_item_list = []
            accelerator_QA_pairs_list = []

            process_index = accelerator.process_index

            rollout_merged_solutions_json_path = os.path.join(args.answer_sheets_dir, "process_index"+str(process_index)+"_"+"Rollout_Merged_Solutions.json")

            rollout_end_solutions_json_path = os.path.join(args.answer_sheets_dir, "process_index"+str(process_index)+"_"+"Rollout_End_Solutions.json")

            final_coco_output_path = os.path.join(args.answer_sheets_dir, "process_index"+str(process_index)+"_"+"Final_Coco_Outputs.json")

            model_all_solutions_json_path = os.path.join(args.answer_sheets_dir, "process_index"+str(process_index)+"_"+"Model_All_Solutions.json")

            all_wrong_data_item_json_path = os.path.join(args.answer_sheets_dir, "Wrong_Data_Items.json")

            question_json_path = os.path.join(args.answer_sheets_dir, "process_index"+str(process_index)+"_"+"Questions.json")

            data_item_json_path = os.path.join(args.answer_sheets_dir, "process_index"+str(process_index)+"_"+"Data_Items.json")

            all_solution_trace_json_path = os.path.join(args.answer_sheets_dir, "process_index"+str(process_index)+"_"+"All_Solution_Trace.json")

            rollout_solution_trace_json_path = os.path.join(args.answer_sheets_dir, "process_index"+str(process_index)+"_"+"Rollout_Solution_Trace.json")

            for path in [
                rollout_merged_solutions_json_path, rollout_end_solutions_json_path, final_coco_output_path, model_all_solutions_json_path,
                question_json_path, data_item_json_path,
                all_solution_trace_json_path, rollout_solution_trace_json_path,
                all_wrong_data_item_json_path
            ]:
                if not os.path.exists(path):
                    with open(path, 'w') as f:
                        json.dump([], f)

            for data_item in tqdm(dataset, desc=f"Proc {process_index}", position=process_index, leave=False, dynamic_ncols=True):
                accelerator_data_item_list.append(data_item)

                problem_id, problem, object_descriptions, gt_solution, gt_answer, image_pil = extract_Grounding_information(data_item, args, evaluator)
                image_path=data_item['image_info']['file_name']

                question_json = {
                    "question id": problem_id,
                    "problem": problem,
                    "object_descriptions": object_descriptions,
                    "model_completion": None,
                    "model_answer": None,
                    "all_model_completions": {},
                    "gold_solution": gt_solution,
                    "gold_answer": gt_answer,
                    "image": image_path,
                }
                accelerator_question_json_list.append(question_json)

                with open(question_json_path, 'r') as f:
                    existing_question_data = json.load(f)
                existing_question_data.append(question_json)
                with open(question_json_path, 'w') as f:
                    json.dump(existing_question_data, f, indent=4)

                try:
                    stopping_id, model_rollout_merged_solutions, model_all_solutions, model_rollout_end_solutions, all_solution_nodes, model_rollout_nodes, all_leaf_nodes = MLLM_search_for_Grounding(
                    args=args, user_question=problem, object_descriptions=object_descriptions, question_id=problem_id, gt_answer=gt_solution, generator=generator, question_image=image_pil, image_path=image_path
                )
                except Exception as e:
                    if 'image_pil' in data_item:
                        del data_item['image_pil']
                    accelerator_wrong_data_item_list.append(data_item)

                    continue

                model_rollout_merged_solutions_dict = {
                    "image_id": data_item['image_info']['file_name'],
                    "solutions": model_rollout_merged_solutions,
                }

                with open(rollout_merged_solutions_json_path, 'r') as f:
                    existing_rollout_merged_solutions_data = json.load(f)
                existing_rollout_merged_solutions_data.append(model_rollout_merged_solutions_dict)
                with open(rollout_merged_solutions_json_path, 'w') as f:
                    json.dump(existing_rollout_merged_solutions_data, f, indent=4)

                model_all_solutions_dict = {
                    "image_id": data_item['image_info']['file_name'],
                    "solutions": model_all_solutions,
                }

                with open(model_all_solutions_json_path, 'r') as f:
                    existing_model_all_solutions_data = json.load(f)
                existing_model_all_solutions_data.append(model_all_solutions_dict)
                with open(model_all_solutions_json_path, 'w') as f:
                    json.dump(existing_model_all_solutions_data, f, indent=4)

                if len(model_rollout_end_solutions) > 0:
                    rollout_end_solutions_dict = {
                        "image_id": data_item['image_info']['file_name'],
                        "solutions": model_rollout_end_solutions,
                    }

                    with open(rollout_end_solutions_json_path, 'r') as f:
                        existing_rollout_end_solutions_data = json.load(f)
                    existing_rollout_end_solutions_data.append(rollout_end_solutions_dict)
                    with open(rollout_end_solutions_json_path, 'w') as f:
                        json.dump(existing_rollout_end_solutions_data, f, indent=4)

                assert len(model_rollout_merged_solutions) > 0, f"No solutions found for question {problem_id}."

                rollout_merged_bboxes_list, rollout_merged_scores_list = zip(*[(solution['bbox'], solution['score']) for solution in model_rollout_merged_solutions])
                rollout_merged_bboxes_list = list(rollout_merged_bboxes_list)
                rollout_merged_scores_list = list(rollout_merged_scores_list)

                bboxes_np = np.array(rollout_merged_bboxes_list)
                converted = np.zeros_like(bboxes_np)
                converted[:, 0] = bboxes_np[:, 0]
                converted[:, 1] = bboxes_np[:, 1]
                converted[:, 2] = bboxes_np[:, 2] - bboxes_np[:, 0]
                converted[:, 3] = bboxes_np[:, 3] - bboxes_np[:, 1]

                rollout_merged_coco_bboxes_list = converted.tolist()

                final_coco_bboxes_list = deepcopy(rollout_merged_coco_bboxes_list)
                final_scores_list = deepcopy(rollout_merged_scores_list)

                if len(model_rollout_end_solutions) > 0:
                    for solution in model_rollout_end_solutions:
                        bbox = solution['bbox']
                        score = solution['score']

                        converted_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                        final_coco_bboxes_list.append(converted_bbox)
                        final_scores_list.append(score)

                final_coco_nms_bboxes, final_nms_scores = apply_nms_coco(final_coco_bboxes_list, final_scores_list, iou_threshold=args.nms_bboxes_iou_threshold, score_threshold=args.nms_score_threshold)

                if len(final_nms_scores) > 0:
                    final_coco_output = {
                        "image_id": data_item['image_info']['file_name'],
                        "bbox": final_coco_nms_bboxes[0],
                        "score": final_nms_scores[0],
                    }
                else:
                    if len(rollout_merged_bboxes_list) > 1:
                        merged_boxes, counts = evaluator._merge_bboxes_with_count(torch.tensor(rollout_merged_coco_bboxes_list), method="iou_cluster", iou_threshold=args.merge_bboxes_iou_threshold)
                        max_index = max(enumerate(counts), key=lambda x: x[1])[0]
                        final_merged_bbox = [float(x.item()) for x in merged_boxes[max_index]]
                    else:
                        final_merged_bbox = rollout_merged_coco_bboxes_list[0]
                    final_coco_output = {
                        "image_id": data_item['image_info']['file_name'],
                        "bbox": final_merged_bbox,
                        "score": 1.0,
                    }

                accelerator_final_coco_output_list.append(final_coco_output)

                with open(final_coco_output_path, 'r') as f:
                    existing_final_coco_output_data = json.load(f)
                existing_final_coco_output_data.append(final_coco_output)
                with open(final_coco_output_path, 'w') as f:
                    json.dump(existing_final_coco_output_data, f, indent=4)

                accelerator_QA_pairs_list.append({
                    "question id": problem_id,

                    "groud_truth": [gt_answer[0], gt_answer[1], gt_answer[0]+gt_answer[2], gt_answer[1]+gt_answer[3]],

                    "prediction": [final_coco_output["bbox"][0], final_coco_output["bbox"][1], final_coco_output["bbox"][0]+final_coco_output["bbox"][2], final_coco_output["bbox"][1]+final_coco_output["bbox"][3]],
                    "img_width": data_item['image_info']['width'],
                    "img_height": data_item['image_info']['height'],
                    "original_id": data_item.get('original_id', -1),
                    "expand_id": data_item.get('expand_id', -1),
                    "model_all_solutions": model_all_solutions,
                })

                all_solution_trace = {"question id": problem_id, "trace": [{"trace": node.solution_trace, "rollout_id": node.rollout_id} for node in all_solution_nodes]}
                with open(all_solution_trace_json_path, 'r') as f:
                    existing_final_solution_data = json.load(f)
                existing_final_solution_data.append(all_solution_trace)
                with open(all_solution_trace_json_path, 'w') as f:
                    json.dump(existing_final_solution_data, f, indent=4)

                rollout_solution_trace = {"question id": problem_id, "trace": [{"trace": node.solution_trace, "rollout_id": i} for i, node in enumerate(model_rollout_nodes)]}
                with open(rollout_solution_trace_json_path, 'r') as f:
                    existing_rollout_solution_data = json.load(f)
                existing_rollout_solution_data.append(rollout_solution_trace)
                with open(rollout_solution_trace_json_path, 'w') as f:
                    json.dump(existing_rollout_solution_data, f, indent=4)

                num_tested += 1

                with open(os.path.join(args.run_outputs_dir, "intermediate_result.txt"), "w") as f:
                    f.write(
                        f"Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}\n"
                    )
                    f.write(
                        f"Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}\n"
                    )

    end_time = time.time()
    accelerator.wait_for_everyone()

    all_data_item_gather_list = gather_object(accelerator_data_item_list)
    all_final_coco_output_gather_list = gather_object(accelerator_final_coco_output_list)
    all_wrong_data_item_gather_list = gather_object(accelerator_wrong_data_item_list)
    all_QA_pairs_gather_list = gather_object(accelerator_QA_pairs_list)

if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--expand_object_descriptions", action="store_true")
    parser.add_argument("--num_rollouts", type=int, default=15)
    parser.add_argument("--text_file_type", type=str, default="json", help="json or parquet")
    parser.add_argument("--max_depth_allowed", type=int, default=5)
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const")
    parser.add_argument("--mcts_num_last_votes", type=int, default=16)
    parser.add_argument("--save_tree", action="store_true")
    parser.add_argument("--debug_print_flag", action="store_true", help="Print debug information for each action.")
    parser.add_argument("--nms_bboxes_iou_threshold", type=float, default=0.5, help="IoU threshold for merging bounding boxes.")
    parser.add_argument("--nms_score_threshold", type=float, default=0.2, help="Score threshold for merging bounding boxes.")
    parser.add_argument("--num_a1_steps", type=int, default=10)
    parser.add_argument("--num_a2_steps", type=int, default=4)
    parser.add_argument("--num_a3_steps", type=int, default=8)
    parser.add_argument("--num_a4_steps", type=int, default=8)
    parser.add_argument("--num_a5_steps", type=int, default=4)
    parser.add_argument("--bbox_type", type=str, choices=["absolute", "relative_1000", "relative_1"], default="absolute")
    parser.add_argument("--merge_bboxes_iou_threshold", type=float, default=0.8)
    parser.add_argument("--bbox2image_iou_threshold", type=float, default=0.2)
    parser.add_argument("--obj_within_bbox_num_threshold", type=float, default=0.5)
    parser.add_argument("--obj_within_image_num_threshold", type=float, default=0.5)
    parser.add_argument("--bbox_extend_ratio", type=float, default=0.7)
    parser.add_argument("--bbox_extend_min_size", type=int, default=28)
    parser.add_argument("--rephrased_description_top_n", type=int, default=3)
    parser.add_argument("--modify_prompts_for_rephrasing", action="store_true")
    parser.add_argument("--enable_potential_score", action="store_true")
    parser.add_argument("--send_bug_email", action="store_true", help="If set, do not send error email.")
    parser.add_argument("--image_root", type=str, default="./data/images", help="Root directory for images.")
    parser.add_argument("--evaluator_type", type=str)
    args = parser.parse_args()
    prompts_dir = os.path.join(args.prompts_root, args.evaluator_type)
    if os.path.exists(prompts_dir + "/transformers_prompt_template.json"):
        args.transformers_prompt_template_path = os.path.join(prompts_dir, "transformers_prompt_template.json")
    args.direct_location_prompt_path = os.path.join(prompts_dir, "location", "direct_location_prompt.txt")
    args.bbox_description_prompt_path = os.path.join(prompts_dir, "description", "bbox_description_prompt.txt")
    args.original_image_description_prompt_path = os.path.join(prompts_dir, "description", "original_image_description_prompt.txt")
    args.location_within_bbox_prompt_path = os.path.join(prompts_dir, "location", "location_within_bbox_prompt.txt")
    args.rephrased_objection_description_prompt_path = os.path.join(prompts_dir, "description", "rephrased_objection_description_prompt.txt")
    args = post_process_args(args)
    print(args)
    save_args(args)
    try:
        main(args)
    except Exception as e:
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        if args.send_bug_email:
            send_error_email(error_msg)
        sys.exit(1)
