
# MedAgent-Pro

This is the official repository for **MedAgent-Pro**, an agentic framework designed for evidence-based, multi-modal medical diagnosis via a reasoning-driven workflow.

## Features

- Hierarchical agentic workflow for medical diagnosis
- Integrated toolset for segmentation, grounding, and visual-language reasoning
- Supports multi-modal diagnostic tasks and dynamic planning




## Usage

### Task-Level Planning

The main code for **task-level planning** can be found in [`Task_level.py`](Task_level.py).

### Case-Level Diagnosis

The main code for **case-level diagnosis** can be found in [`Case_level.py`](Case_level.py).

### Configuring VLMs

Before running the code, you need to set up your **OpenAI API key** or replace the default VLM in the `Planner` agent and qualitative analysis module with other VLMs defined in the `Decider` folder.

```python
OPENAI_API_KEY = "your_api_key"
```

---

## Preparing Diagnostic Tasks

Refer to the **Glaucoma** example for preparing diagnostic configurations.

You should prepare two JSON files:

1. `task.json` – describes the diagnostic task.
2. `toolset.json` – defines the available tools.

### Toolset Format

All tools should follow this function signature:

```python
Function(image_path, save_dir, save_name)
```

* `image_path`: A string or a list of image paths.
* `save_dir`: Directory where the outputs will be saved.
* `save_name`: Filename for the saved result.

---

## Example Structure

```
MedAgent-Pro/
├── Glaucoma/
│   ├── task.json
│   ├── toolset.json
│   ├── ...
├── Task_level.py
├── Case_level.py
├── Decider/
└── ...
```


## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{wang2025medagent,
  title={MedAgent-Pro: Towards Evidence-Based Multi-Modal Medical Diagnosis via Reasoning Agentic Workflow},
  author={Wang, Ziyue and Wu, Junde and Cai, Linghan and Low, Chang Han and Yang, Xihong and Li, Qiaxuan and Jin, Yueming},
  journal={arXiv preprint arXiv:2503.18968},
  year={2025}
}
```



