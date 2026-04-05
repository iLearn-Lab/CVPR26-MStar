import os, json, re
from typing import List, Dict, Tuple, Union, Optional
from collections import defaultdict
import random
from fuzzywuzzy import fuzz, process
import re
from markdown_it import MarkdownIt
import torch
from torchvision.ops.boxes import box_iou
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Evaluator:
    def __init__(self) -> None:
        self.answer_marker = "answer is"

    def _is_number(self, s) -> Tuple[bool, str]:
        try:
            res = float(s)
            return True, str(res)
        except:
            pass
        try:
            import unicodedata

            res = unicodedata.numeric(s)
            return True, str(res)
        except:
            pass
        return False, None

    def validate_completion(self, completion: str) -> bool:
        if self.answer_marker.lower() in completion.lower():
            return True

        return False

    def isolate_answer(self, text: str):
        if text is None:
            return None

        assert isinstance(text, str)
        text = text.lower()
        split_ans = text.split(self.answer_marker.lower())
        if len(split_ans) > 1:
            ans = split_ans[-1].replace(":", "").strip()
            extract_ans_temp = ans.split(".\n")[0].strip()
            if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == ".":
                extract_ans = extract_ans_temp[0:-1]
            else:
                extract_ans = extract_ans_temp
            extract_ans = extract_ans.strip().strip("\n")
            return extract_ans
        else:
            return text

    def determine_if_there_is_a_split_mask_present(self, preds: str):
        assert isinstance(preds, str)

        preds = preds.split(self.answer_marker)
        answer_flag = True if len(preds) > 1 else False
        return answer_flag

    def find_most_confident_answer(self, completions: List[str], prior_weights: List[float] = None):
        if completions is None or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list)
        answer2ids = defaultdict(list)
        answer_flag_dict = defaultdict(list)
        for id, c in enumerate(completions):
            try:
                answer_flag = self.determine_if_there_is_a_split_mask_present(c)

                model_answer = self.extract_answer_from_model_completion(c)
                has_existed = False
                for existing_answer in answer2completions.keys():
                    if self.check_answers_equiv(model_answer, existing_answer):
                        assert not has_existed
                        has_existed = True
                        answer2completions[existing_answer].append(c)
                        answer2ids[existing_answer].append(id)
                        answer_flag_dict[existing_answer].append(answer_flag)
                if not has_existed:
                    answer2completions[model_answer].append(c)
                    answer2ids[model_answer].append(id)
                    answer_flag_dict[model_answer].append(answer_flag)
            except:
                pass

        assert len(answer2completions.keys()) > 0, "There are no valid completions."

        try:
            for answer, answer_flag_list in answer_flag_dict.items():
                assert all([isinstance(answer_flag, bool) for answer_flag in answer_flag_list])
        except:
            print(answer_flag_dict)
            raise AssertionError("Values in answer_flag_list must be of bool type")

        marker_flag_dict = {}
        for answer, answer_flag_list in answer_flag_dict.items():
            marker_flag_dict[answer] = any([answer_flag for answer_flag in answer_flag_list])
        if any(marker_flag_dict.values()):
            for answer, marker_flag in marker_flag_dict.items():
                if not marker_flag:
                    del answer2completions[answer]
                    del answer2ids[answer]
                else:
                    pass
        else:
            pass

        if prior_weights is not None:
            assert len(completions) == len(prior_weights)
            completion2count = {}
            for answer, answer_completions in answer2completions.items():
                count = len(answer_completions)
                for answer_completion in answer_completions:
                    completion2count[answer_completion] = count

            completion2score = {}
            for id, (completion, count) in enumerate(completion2count.items()):
                prior_weight = prior_weights[id]
                score = prior_weight * (count / len(completions))
                completion2score[completion] = score

            most_confident_completion = max(completion2score.keys(), key=lambda x: completion2score[x])

            return (
                self.extract_answer_from_model_completion(most_confident_completion),
                most_confident_completion,
                completions.index(most_confident_completion),
                completion2score[most_confident_completion],
            )
        else:
            most_confident_answer = max(answer2completions.keys(), key=lambda x: len(answer2completions[x]))
            assert (
                len(answer2completions[most_confident_answer]) > 0
            ), "There are no completions for the most confident answer."
            confidence = len(answer2completions[most_confident_answer]) / len(completions)
            assert confidence > 0

            index, most_confident_completions = max(enumerate(answer2completions[most_confident_answer]), key=lambda x: len(x[1]))

            return (
                most_confident_answer,
                most_confident_completions,
                answer2ids[most_confident_answer][index],
                confidence,
            )

    def stochastic_select_answer(self, completion2score, answer2completions, completions):
        answer2score = {}
        answer_counts = {}
        for completion, score in completion2score.items():
            answer = self.extract_answer_from_model_completion(completion)
            if answer in answer2score:
                answer2score[answer] += score
                answer_counts[answer] += 1
            else:
                answer2score[answer] = score
                answer_counts[answer] = 1

        for answer in answer2score:
            answer2score[answer] /= answer_counts[answer]

        top_answers = sorted(answer2score.items(), key=lambda x: x[1], reverse=True)[:1]
        answers, scores = zip(*top_answers)
        total_score = sum(scores)
        try:
            probabilities = [score / total_score for score in scores]
            selected_answer = random.choices(answers, weights=probabilities, k=1)[0]
        except:
            selected_answer = random.choices(answers, k=1)[0]

        most_confident_completion = answer2completions[selected_answer][0]
        completion_index = completions.index(most_confident_completion)
        confidence = answer2score[selected_answer]

        return selected_answer, most_confident_completion, completion_index, confidence

    def stochastic_calculate_completion_scores(self, prior_weights, answer2completions):
        completion2count = {}
        for answer, comps in answer2completions.items():
            count = len(comps)
            for comp in comps:
                completion2count[comp] = count

        completion2score = {}
        for idx, comp in enumerate(completion2count.keys()):
            weight = prior_weights[idx] if prior_weights is not None else 1
            score = weight * completion2count[comp]
            completion2score[comp] = score
        return completion2score

    def stochastic_select_response(self, completion2score, completions):
        sorted_completions = sorted(completion2score.items(), key=lambda x: x[1], reverse=True)[:1]
        completions, scores = zip(*sorted_completions)
        total_score = sum(scores)
        try:
            probabilities = [score / total_score for score in scores]
            sampled_completion = random.choices(completions, weights=probabilities, k=1)[0]
        except:
            sampled_completion = random.choices(completions, k=1)[0]
        confidence = completion2score[sampled_completion]
        most_confident_answer = self.extract_answer_from_model_completion(sampled_completion)
        id_of_most_confident = completions.index(sampled_completion)
        return most_confident_answer, sampled_completion, id_of_most_confident, confidence

    def stochastic_find_most_confident_answer(
        self,
        completions: List[str],
        prior_weights: List[float] = None,
    ):
        if not completions or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list)
        for idx, comp in enumerate(completions):
            try:
                answer = self.extract_answer_from_model_completion(comp)
                answer2completions[answer].append(comp)
            except:
                continue

        if not answer2completions:
            return None, None, None, None

        completion2score = self.stochastic_calculate_completion_scores(prior_weights, answer2completions)

        most_confident_answer, sampled_completion, id_of_most_confident, confidence = self.stochastic_select_response(
            completion2score, completions
        )
        return most_confident_answer, sampled_completion, id_of_most_confident, confidence

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        raise NotImplementedError

    def extract_answer_from_gold_solution(self, solution: str) -> str:
        raise NotImplementedError

    def extract_answer_from_model_completion(self, completion: str) -> str:
        raise NotImplementedError

def clean_text(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)

    def process_word(word):
        return word.lower() if re.match(r'^[a-zA-Z]+$', word) else word

    cleaned_text = ' '.join(process_word(word) for word in cleaned_text.split())

    return cleaned_text

class RefCOCOEvaluator(Evaluator):
    def __init__(self, iou_thresholds: List[float] = None) -> None:
        super().__init__()

        if iou_thresholds is None:
            self.iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
        else:
            self.iou_thresholds = iou_thresholds

        self.gt_data = []
        self.pred_data = []
        self.image_info = []
        self.image_id_map = {}
        self.next_image_id = 1
        self.next_ann_id = 1

        self.vectorizer = TfidfVectorizer()

    def _extract_json_blocks(self, json_text_list):
        md = MarkdownIt()
        dict_list = []
        filtered_dict_list = []
        for item in json_text_list:
            tokens = md.parse(item)
            for token in tokens:
                if token.type == 'fence' and token.info == 'json':
                    try:
                        dict_group = json.loads(token.content)
                        for dict_iter in dict_group:
                            if isinstance(dict_iter, dict):
                                dict_list.append(dict_iter)
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON block after attempting to fix it: {e}")
                        print(f"Original content: {token.content}")

        for iter in dict_list:
            if "bbox_2d" in iter.keys() and "label" in iter.keys():
                bbox_2d = iter["bbox_2d"]
                if isinstance(bbox_2d, list) and len(bbox_2d) == 4 and all(isinstance(i, int) for i in bbox_2d):
                    filtered_dict_list.append(iter)
        return filtered_dict_list

    def _extract_bold_text_with_markdown_it(self, markdown_string):
        md = MarkdownIt()
        tokens = md.parse(markdown_string)

        bold_texts = []

        def extract_from_tokens(token_list):
            i = 0
            while i < len(token_list):
                token = token_list[i]

                if token.type == 'strong_open':
                    j = i + 1
                    content = ""
                    while j < len(token_list) and token_list[j].type != 'strong_close':
                        if token_list[j].type == 'text':
                            content += token_list[j].content
                        elif token_list[j].children:
                            content += extract_text_from_children(token_list[j].children)
                        j += 1

                    if content.strip():
                        bold_texts.append(content.strip())
                    i = j + 1

                elif hasattr(token, 'children') and token.children:
                    extract_from_tokens(token.children)
                    i += 1
                else:
                    i += 1

        def extract_text_from_children(children):
            text = ""
            for child in children:
                if child.type == 'text':
                    text += child.content
                elif hasattr(child, 'children') and child.children:
                    text += extract_text_from_children(child.children)
            return text

        extract_from_tokens(tokens)
        return bold_texts

    def _extract_bold_text_simple(self, markdown_string):
        md = MarkdownIt()

        html = md.render(markdown_string)

        import re
        pattern = r'<strong>(.*?)</strong>'
        matches = re.findall(pattern, html, re.DOTALL)

        clean_texts = []
        for match in matches:
            clean_text = re.sub(r'<[^>]+>', '', match).strip()
            if clean_text:
                clean_texts.append(clean_text)

        return clean_texts

    def _merge_bboxes_with_count(self, bboxes, method="average", iou_threshold=0.5):
        if method == "average":
            merged_box = bboxes.mean(dim=0, keepdim=True)
            counts = [len(bboxes)]
            return merged_box, counts

        elif method == "iou_cluster":
            merged_boxes = []
            counts = []

            used = torch.zeros(len(bboxes), dtype=torch.bool)

            for i in range(len(bboxes)):
                if used[i]:
                    continue

                current_group = [bboxes[i]]
                used[i] = True

                for j in range(i + 1, len(bboxes)):
                    if used[j]:
                        continue

                    iou = box_iou(current_group[-1].unsqueeze(0), bboxes[j:j+1]).item()
                    if iou > iou_threshold:
                        current_group.append(bboxes[j])
                        used[j] = True

                group_tensor = torch.stack(current_group)
                merged_box = group_tensor.mean(dim=0)
                merged_boxes.append(merged_box)
                counts.append(len(current_group))

            return torch.stack(merged_boxes), counts

        else:
            raise ValueError("Unsupported merge method: {}".format(method))

    def _inspect_and_clip_bbox_with_iou(self, bbox, image_size, iou_threshold=0.3):
        x1, y1, x2, y2 = bbox
        width, height = image_size

        inter_x1 = max(0, min(x1, width - 1))
        inter_y1 = max(0, min(y1, height - 1))
        inter_x2 = max(0, min(x2, width - 1))
        inter_y2 = max(0, min(y2, height - 1))

        if x2 < 0 or x1 >= width or y2 < 0 or y1 >= height:
            return [0, 0, 0, 0], True

        clipped_bbox = [inter_x1, inter_y1, inter_x2, inter_y2]

        def area(box):
            x1, y1, x2, y2 = box
            return max(0, x2 - x1) * max(0, y2 - y1)

        inter_area = area([inter_x1, inter_y1, inter_x2, inter_y2])
        bbox_area = area([x1, y1, x2, y2])

        if bbox_area == 0:
            return clipped_bbox, True

        overlap_ratio = inter_area / bbox_area

        wrong_terminal_node_flag = overlap_ratio < iou_threshold

        return clipped_bbox, wrong_terminal_node_flag

    def _str_to_bool(self, bool_str):
        if isinstance(bool_str, str):
            s_lower = bool_str.strip().lower()
            if s_lower == "true":
                return True
            elif s_lower == "false":
                return False
        raise ValueError(f"Cannot convert '{bool_str}' to bool.")

    def _extract_code_blocks_from_markdown(self, md_text):
        md = MarkdownIt()
        tokens = md.parse(md_text)

        code_blocks = []
        for token in tokens:
            if token.type == 'fence':
                code_blocks.append(token.content.strip())

        code_content_list = []
        for idx, block in enumerate(code_blocks):
            code_content_list.append(block)

        return code_content_list

    def _calculate_description_overlap_score(self, descriptions_list):
        if not descriptions_list or len(descriptions_list) < 2:
            return 0.0

        def method1_word_overlap():
            word_sets = []
            for desc in descriptions_list:
                words = set(desc.lower().split())
                word_sets.append(words)

            intersection = word_sets[0]
            union = word_sets[0].copy()

            for word_set in word_sets[1:]:
                intersection = intersection.intersection(word_set)
                union = union.union(word_set)

            if len(union) == 0:
                return 0.0
            return len(intersection) / len(union)

        def method2_char_similarity():
            from difflib import SequenceMatcher

            scores = []
            n = len(descriptions_list)

            for i in range(n):
                for j in range(i + 1, n):
                    similarity = SequenceMatcher(None, descriptions_list[i], descriptions_list[j]).ratio()
                    scores.append(similarity)

            return sum(scores) / len(scores) if scores else 0.0

        def method3_fuzzy_similarity():
            try:
                from rapidfuzz import fuzz

                scores = []
                n = len(descriptions_list)

                for i in range(n):
                    for j in range(i + 1, n):
                        similarity = fuzz.token_sort_ratio(descriptions_list[i], descriptions_list[j]) / 100.0
                        scores.append(similarity)

                return sum(scores) / len(scores) if scores else 0.0
            except ImportError:
                return method2_char_similarity()

        def method4_combined():
            word_score = method1_word_overlap()
            char_score = method3_fuzzy_similarity()

            return 0.6 * word_score + 0.4 * char_score

        return method4_combined()

    def _calculate_similarity(self, texts):
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix

    def _find_most_similar(self, texts, m):
        if m > len(texts):
            raise ValueError("m cannot be greater than the total number of texts")

        similarity_matrix = self._calculate_similarity(texts)

        n = len(texts)
        similarities = []

        from itertools import combinations
        best_combination = None
        best_score = -1

        for combination in combinations(range(n), m):
            total_similarity = 0
            pair_count = 0

            for i in range(len(combination)):
                for j in range(i + 1, len(combination)):
                    idx1, idx2 = combination[i], combination[j]
                    total_similarity += similarity_matrix[idx1][idx2]
                    pair_count += 1

            if pair_count > 0:
                avg_similarity = total_similarity / pair_count
                if avg_similarity > best_score:
                    best_score = avg_similarity
                    best_combination = combination

        return best_combination, best_score

    def _get_similar_texts(self, texts, top_n):
        indices, score = self._find_most_similar(texts, top_n)
        similar_texts = [texts[i] for i in indices]
        return similar_texts, score

    def _get_numeric_image_id(self, image_id_str: str) -> int:
        if image_id_str not in self.image_id_map:
            self.image_id_map[image_id_str] = self.next_image_id
            self.next_image_id += 1
        return self.image_id_map[image_id_str]

    def add_ground_truth(self, sample_data: Dict):
        image_id_str = sample_data['image']
        image_id = self._get_numeric_image_id(image_id_str)

        image_info = sample_data.get('image_info', {})
        self.image_info.append({
            'id': image_id,
            'file_name': image_info.get('file_name', image_id_str),
            'width': image_info.get('width', 640),
            'height': image_info.get('height', 480),
            'license': image_info.get('license', 1),
            'coco_url': image_info.get('coco_url', ''),
            'date_captured': image_info.get('date_captured', ''),
            'flickr_url': image_info.get('flickr_url', '')
        })

        for ann in sample_data['anns']:
            bbox = ann['bbox']

            if len(bbox) == 4:
                x, y, w, h = bbox
                area = w * h
            else:
                continue

            self.gt_data.append({
                'id': self.next_ann_id,
                'image_id': image_id,
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': area,
                'iscrowd': ann.get('iscrowd', 0),
                'segmentation': ann.get('segmentation', [])
            })
            self.next_ann_id += 1

    def add_ground_truth_batch(self, sample_data_list: List[Dict]):
        for sample_data in sample_data_list:
            self.add_ground_truth(sample_data)

    def add_prediction(self, image_id: str, bbox: List[float], score: Optional[float] = None):
        image_id = self._get_numeric_image_id(image_id)

        if score is None:
            score = 1.0

        self.pred_data.append({
            'image_id': image_id,
            'category_id': 1,
            'bbox': bbox,
            'score': score
        })

    def add_predictions_batch(self, predictions: List[Dict]):
        for pred in predictions:
            self.add_prediction(
                image_id=pred['image_id'],
                bbox=pred['bbox'],
                score=pred.get('score', None)
            )

    def add_predictions_multiple(self, image_id_str: str, bboxes: List[List[float]], scores: Optional[List[float]] = None):
        image_id = self._get_numeric_image_id(image_id_str)

        if scores is None:
            scores = [1.0 - i * 0.01 for i in range(len(bboxes))]

        if len(scores) != len(bboxes):
            raise ValueError(f"Length of scores ({len(scores)}) does not match length of bboxes ({len(bboxes)})")

        for bbox, score in zip(bboxes, scores):
            self.pred_data.append({
                'image_id': image_id,
                'category_id': 1,
                'bbox': bbox,
                'score': score
            })

    def add_predictions_batch_multiple(self, predictions: List[Dict]):
        for pred in predictions:
            self.add_predictions_multiple(
                image_id_str=pred['image_id'],
                bboxes=pred['bboxes'],
                scores=pred.get('scores', None)
            )

    def _create_coco_gt_file(self) -> str:
        unique_images = {}
        for img in self.image_info:
            if img['id'] not in unique_images:
                unique_images[img['id']] = img

        coco_gt = {
            'info': {
                'description': 'RefCOCO Dataset for Grounding Task',
                'version': '1.0',
                'year': 2024,
                'contributor': 'RefCOCO Evaluator',
                'date_created': '2024-07-21'
            },
            'licenses': [
                {'id': 1, 'name': 'Unknown', 'url': ''}
            ],
            'images': list(unique_images.values()),
            'annotations': self.gt_data,
            'categories': [
                {'id': 1, 'name': 'object', 'supercategory': 'object'}
            ]
        }

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(coco_gt, temp_file, indent=2)
        temp_file.close()

        return temp_file.name

    def _create_coco_pred_file(self) -> str:
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.pred_data, temp_file, indent=2)
        temp_file.close()

        return temp_file.name

    def print_pycocotools_results(self, results):
        if not results:
            print("No results to display. Please call evaluate() first to obtain evaluation results.")
            return

        print("RefCOCO Grounding Evaluation Results (using pycocotools):")
        print("=" * 60)

        print("Custom IoU Thresholds:")
        for iou_threshold in self.iou_thresholds:
            if f'AP@{iou_threshold:.2f}' in results:
                ap = results[f'AP@{iou_threshold:.2f}']
                print(f"  AP@{iou_threshold:.2f}: {ap:.4f}")

        print(f"  mAP (custom): {results.get('mAP', 0):.4f}")

        print("\nCOCO Standard Metrics:")
        print(f"  AP@0.5:0.95: {results.get('AP@0.5:0.95', 0):.4f}")
        print(f"  AP@0.5: {results.get('AP@0.5', 0):.4f}")
        print(f"  AP@0.75: {results.get('AP@0.75', 0):.4f}")

    def refcoco_bbox_rec_process_result(self, qa_pair):
        groud_truth = qa_pair['groud_truth']
        prediction = qa_pair['prediction']
        img_width = qa_pair['img_width']
        img_height = qa_pair['img_height']

        def convert_bbox_to_relative(bbox, img_width, img_height):
            x1, y1, x2, y2 = bbox
            x1_rel = x1 / img_width
            y1_rel = y1 / img_height
            x2_rel = x2 / img_width
            y2_rel = y2 / img_height
            return [x1_rel, y1_rel, x2_rel, y2_rel]

        relative_gt_bbox = convert_bbox_to_relative(groud_truth, img_width, img_height)
        relative_pred_bbox = convert_bbox_to_relative(prediction, img_width, img_height)

        if 'model_all_solutions' in qa_pair:
            relative_all_solutions = []
            for solution in qa_pair['model_all_solutions']:
                abs_bbox = solution['bbox']
                score = solution.get('score', 1.0)
                rel_bbox = convert_bbox_to_relative(abs_bbox, img_width, img_height)
                relative_all_solutions.append({'bbox': rel_bbox, 'score': score})
            qa_pair['model_all_solutions'] = relative_all_solutions

        qa_pair['groud_truth'] = relative_gt_bbox
        qa_pair['prediction'] = relative_pred_bbox
        return qa_pair

    def compute_iou(self, box1, box2):
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - intersection_area

        iou = intersection_area / union_area

        return iou

    def compute_accuracy(self, box1, box2, threshold=0.5):
        iou = self.compute_iou(box1, box2)
        return iou >= threshold

    def compute_center_accuracy(self, box1, box2):
        center_x = (box2[0] + box2[2]) / 2
        center_y = (box2[1] + box2[3]) / 2

        return box1[0] <= center_x <= box1[2] and box1[1] <= center_y <= box1[3]

    def refcoco_bbox_rec_aggregation_result(self, qa_pairs_list, metric_list=["ACC@0.1", "ACC@0.3", "ACC@0.5", "ACC@0.7", "ACC@0.9", "Center_ACC"]):
        scorers = {
            "ACC@0.1": lambda x, y: self.compute_accuracy(x, y, 0.1),
            "ACC@0.3": lambda x, y: self.compute_accuracy(x, y, 0.3),
            "ACC@0.5": lambda x, y: self.compute_accuracy(x, y, 0.5),
            "ACC@0.7": lambda x, y: self.compute_accuracy(x, y, 0.7),
            "ACC@0.9": lambda x, y: self.compute_accuracy(x, y, 0.9),
            "Center_ACC": self.compute_center_accuracy,
        }

        results_dict = {m: [] for m in metric_list if m in scorers}
        qa_pairs_with_correctness_list = []
        for qa_pair in qa_pairs_list:
            qa_pair = self.refcoco_bbox_rec_process_result(qa_pair)
            gt_bbox = qa_pair["groud_truth"]
            pred_bbox = qa_pair["prediction"]

            for metric in metric_list:
                if metric not in scorers:
                    continue
                score = scorers[metric](gt_bbox, pred_bbox)
                results_dict[metric].append(score)

                qa_pair[f"is_correct_{metric}"] = score
            qa_pairs_with_correctness_list.append(qa_pair)
        print("\n=========== LMMS Aggregated Results ===========\n")
        for metric in metric_list:
            if len(results_dict[metric]) > 0:
                results_dict[metric] = sum(results_dict[metric]) / len(results_dict[metric])
                print(f"Aggregated {metric} score: {results_dict[metric]}")
            else:
                results_dict[metric] = 0
                print(f"Aggregated {metric} score: 0 (no samples)")

        summary_lines = []
        for metric in metric_list:
            summary_lines.append(f"Aggregated {metric} score: {results_dict[metric]}")
        results_txt = "\n".join(summary_lines)
        return results_txt, qa_pairs_with_correctness_list
