<div align="center">
<h2 align="center">
   ⭐ <b>M-Star: Breaking the Regional Perception Bottleneck of Multimodal Large Language Models via External Reasoning Framework</b>
</h2>
<div>
<a target="_blank">Jinrong&#160;Zhang</a><sup>1</sup>,
<a target="_blank">Zhaoyang&#160;Xu</a><sup>1</sup>,
<a target="_blank">Xusheng&#160;He</a><sup>1</sup>,
<a target="_blank">Xinrui&#160;Li</a><sup>1</sup>,
<a target="_blank">Na&#160;Zheng</a><sup>1</sup>,
<a target="_blank">Jianlong&#160;Wu</a><sup>1&#9993</sup>
</div>
<br />
<sup>&#9993&#160;</sup>Corresponding author&#160;&#160;</span>
<br/>
<div align="center">
    <img src="https://img.shields.io/badge/Conference-CVPR%202026-blue" alt="CVPR 2026">
</div>
</div>

## :new: Updates

- [04/2026] :fire: We update the **Action Design** of M-Star. The reasoning action space is restructured into five complementary action types (A1–A5) that drive the Monte Carlo Tree Search process. Each action targets a specific perceptual or linguistic bottleneck:

- [04/2026] :fire: Code for M-Star visual grounding framework released.

## :rocket: M-Star

M-Star is a visual grounding framework that applies Monte Carlo Tree Search (MCTS) over a Multimodal Large Language Model (MLLM) to iteratively refine bounding-box predictions. Instead of a single-shot prediction, the model explores a tree of reasoning actions — describing regions, rephrasing object queries, cropping and re-localizing — and aggregates the most supported bounding boxes across all rollouts.

### How It Works

Each grounding query is the root of an MCTS tree. At every node the model chooses one of five action types:

| ID | Action | Description |
|----|--------|-------------|
| A1 | **Direct Location** | Predict a bounding box directly from the current view |
| A2 | **BBox Description** | Describe the content inside the current bounding box to verify it contains the target |
| A3 | **Original Image Description** | Describe the full image to determine whether the target exists |
| A4 | **Location Within BBox** | Crop to the current bounding box and re-localize the target inside it |
| A5 | **Rephrased Object Description** | Generate an alternative phrasing of the referring expression to reduce ambiguity |

The tree is searched for `--num_rollouts` rollouts up to depth `--max_depth_allowed`. Terminal bounding boxes from all rollouts are merged via IoU clustering and NMS to produce the final prediction.

## :wrench: Requirements

- Python 3.11
- CUDA 12.6
- 4 × NVIDIA A100 40GB (or equivalent VRAM)

All Python dependencies are pinned in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

> **Note** — `transformers >= 4.49.0` is required for Qwen2.5-VL support.  
> `fuzzywuzzy` is paired with `python-Levenshtein` to suppress the speed warning at runtime.

## :file_folder: Data Preparation

Download COCO 2014 images and the RefCOCO / RefCOCO+ / RefCOCOg annotation JSON files. Arrange them as follows:

```
data/
└── coco/
    ├── train2014/          # COCO train2014 images
    └── RefCOCO/
        ├── refcoco_val.json
        ├── refcoco_testA.json
        └── refcoco_testB.json
```

## :robot: Model Preparation

Download a supported checkpoint and place it under `checkpoints/`. The default configuration uses Qwen2.5-VL-7B-Instruct:

```
checkpoints/
└── Qwen/
    └── Qwen2.5-VL-7B-Instruct/
```

Any HuggingFace-compatible vision-language model can be substituted via `--model_ckpt`.

## :video_game: Running

### Quick start (4 GPUs, RefCOCO val split)

```bash
export MASTER_PORT=29500
bash scripts/run_MLLM_Grounding_accelerator_v2.sh
```

### Full argument reference

```bash
accelerate launch --num_processes <N> --main_process_port <PORT> \
    run_src/MLLM_do_Grounding_accelerator.py \
    --seed 42 \
    --api transformers \
    --model_ckpt <path/to/checkpoint> \
    --text_file_type json \
    --data_root data/coco \
    --image_root data/coco/train2014 \
    --dataset_name RefCOCO \
    --evaluator_type RefCOCO \
    --test_json_filename refcoco_val \
    --note <run_tag> \
    --max_depth_allowed 4 \
    --num_rollouts 16 \
    --max_tokens 2048 \
    --num_a1_steps 10 \
    --num_a2_steps 4 \
    --num_a3_steps 4 \
    --num_a4_steps 8 \
    --num_a5_steps 4
```

## :bar_chart: Outputs

Results are written to `run_outputs/<dataset>/<model>/<timestamp>---[<note>]/`:

```
answer_sheets/
├── process_index<N>_Questions.json
├── process_index<N>_Rollout_Merged_Solutions.json
├── process_index<N>_Rollout_End_Solutions.json
├── process_index<N>_Final_Coco_Outputs.json
├── process_index<N>_All_Solution_Trace.json
└── process_index<N>_Rollout_Solution_Trace.json
intermediate_result.txt
args.json
```

## :memo: Prompt Templates

Prompts are stored under `prompts/<evaluator_type>/` and loaded automatically at runtime. To adapt M-Star to a new dataset, create a new subdirectory with the same file structure:

```
prompts/<YourDataset>/
├── transformers_prompt_template.json
├── location/
│   ├── direct_location_prompt.txt
│   └── location_within_bbox_prompt.txt
└── description/
    ├── bbox_description_prompt.txt
    ├── original_image_description_prompt.txt
    └── rephrased_objection_description_prompt.txt
```

## :hugs: Citation

If you find this work useful for your research, please kindly cite our paper:

```
@inproceedings{zhang2026breaking,
  title={Breaking the Regional Perception Bottleneck of Multimodal Large Language Models via External Reasoning Framework},
  author={Zhang, Jinrong and Xu, Zhaoyang and He, Xusheng and Li, Xinrui and Zheng, Na and Wu, Jianlong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
