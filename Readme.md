# RGCL: Improving Hateful Meme Detection through Retrieval-Guided Contrastive Learning

This is the official repository for two research papers:
- **RGCL**: Improving Hateful Meme Detection through Retrieval-Guided Contrastive Learning (ACL 2024)
- **RA-HMD**: Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection (EMNLP 2025 Oral)

---

**Resources:**
- [RGCL Paper](https://aclanthology.org/2024.acl-long.291.pdf) (ACL 2024)
- [RA-HMD Paper](https://aclanthology.org/2025.emnlp-main.1215/) (EMNLP 2025)
- [Project Page](https://rgclmm.github.io/)


## Updates
- **[21/08/2025]** 🔥 RA-HMD accepted to EMNLP 2025 Main Conference. Full code will be released shortly.
- **[27/03/2025]** 🔥 RA-HMD Stage 1 code released. Available in the [LLAMA-FACTORY@a88f610](https://github.com/JingbiaoMei/LLaMA-Factory-LMM-RGCL/tree/a88f610e9fa46d1ef1669c5dbc39ee9008f95c21) submodule.
- **[18/02/2025]** 🔥 RA-HMD paper released on [arXiv](https://arxiv.org/abs/2502.13061).
- **[29/10/2024]** 🔥 Initial codebase release.
- **[10/08/2024]** 🔥 RGCL presented at ACL 2024 Main Conference.


# RGCL

## Usage

### Environment Setup

Create and activate a new conda environment:
```shell
conda create -n RGCL python=3.10 -y
conda activate RGCL
```

Install PyTorch:
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

Install FAISS:
```shell
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl -y
```

Install additional requirements:
```shell
pip install -r requirements.txt
```


## Dataset Preparation

### Image Data
Place all images in the `./data/image/dataset_name/All` directory structure.

**Examples:**
- `./data/image/FB/All/12345.png`
- `./data/image/HarMeme/All/`
- `./data/image/Propaganda/All/`

### Annotation Data
Place JSONL annotation files in the `./data/gt/dataset_name` directory.

### Generate CLIP Embeddings
To optimize training efficiency, CLIP embeddings are pre-generated to avoid redundant computation during training.

```shell
python3 src/utils/generate_CLIP_embedding_HF.py --dataset "FB"
python3 src/utils/generate_CLIP_embedding_HF.py --dataset "HarMeme"
```

### Generate ALIGN Embeddings
```shell
python3 src/utils/generate_ALIGN_embedding_HF.py --dataset "FB"
python3 src/utils/generate_ALIGN_embedding_HF.py --dataset "HarMeme"
```

### Generate Sparse Retrieval Index (Optional)
We provide Sparse Retrieval as a comparison to the proposed dynamic retrieval method in RGCL.
#### VinVL Bounding Box Prediction 
Object detection bounding boxes were generated using VinVL. To facilitate reproducibility, we provide pre-extracted bounding box predictions for the HatefulMemes dataset:

**Download:** [HuggingFace Dataset](https://huggingface.co/datasets/Jingbiao/rgcl-sparse-retrieval/tree/main)  


## Training and Evaluation

Run the following command to start training and evaluation:
```shell
bash scripts/experiments.sh
```

### Common Issues

**Training Hangs or Freezes:**  
If training appears to be stuck, this is often related to the FAISS installation. Ensure FAISS-GPU is properly installed following the environment setup instructions above. 




# RA-HMD


## RA-HMD Stage 1 Code
The Stage 1 training code for RA-HMD is now available. This release is based on an updated version of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (newer than the paper implementation) to support Qwen2.5-VL training.





### Environment Setup

Clone the repository and set up the environment:
```shell
git clone https://github.com/JingbiaoMei/RGCL.git
cd RGCL

# Initialize and update submodules
git submodule update --init --recursive

# Navigate to the LLaMA-Factory submodule
cd RA-HMD/LLAMA-FACTORY
conda create -n llamafact python=3.10 -y
conda activate llamafact
pip install -e ".[torch,metrics,deepspeed,liger-kernel,bitsandbytes,qwen]"
pip install torchmetrics wandb easydict
pip install qwen_vl_utils torchvision
# Install FAISS
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl -y
```


### Dataset
**Dataset:**  
Original datasets and LLaMA-Factory-compatible formats are available on [HuggingFace](https://huggingface.co/datasets/Jingbiao/RA-HMD). 


The dataset follows the ShareGPT-style conversational format required by LLaMA-Factory. Below is an example entry from `data/dataset_info.json`.
You must place the actual JSON file (e.g., sharegpt_FB_train_instructblip.json) into the path specified under "file_name":
```json
  "hatefulmemes": {
    "file_name": "gt/FB/sharegpt_FB_train_instructblip.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
```

### Reproduced Qwen2.5-VL Results

In addition to the Qwen2-VL results reported in the paper, we provide evaluation results using **Qwen2.5-VL-7B** on the same benchmarks.

| Model             | HatefulMemes (Acc) | HatefulMemes (F1) | MAMI (Acc) | MAMI (F1) | PrideMM (Acc) | PrideMM (F1) |
|-------------------|-------------------:|------------------:|------------:|-----------:|---------------:|--------------:|
| **RA-HMD (Qwen2.5-VL-7B)** | **80.8** | **80.1** | **81.0** | **81.0** | **78.0** | **77.8** |

The rubric-based LLM-as-Judge evaluation on HatefulMemes yields a score of **5.4** for RA-HMD (Qwen2.5-VL-7B) explanations.

**Summary:** RA-HMD (Qwen2.5-VL-7B) achieves comparable performance to RA-HMD (Qwen2-VL-7B), with minor variations across different metrics.


### December 2025 Update

**LLaMA-Factory Ver2025.12** is now available with support for the latest models including InternVL-3, Qwen3-VL, and more. We provide a ported version of RA-HMD Stage 1 training code compatible with this release.

#### Installation

```shell
git clone https://github.com/JingbiaoMei/RGCL.git
cd RGCL

# Initialize and update submodules
git submodule update --init --recursive

# Navigate to the LLaMA-Factory submodule
cd RA-HMD/LLAMA-FACTORY
git checkout Ver202512
conda create -n llamafact202512 python=3.10 -y
conda activate llamafact202512
pip install -e ".[torch,metrics,deepspeed,liger-kernel,bitsandbytes,qwen]"
pip install torchmetrics wandb easydict qwen_vl_utils torchvision
# Install FAISS
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl -y
```

#### Troubleshooting

**Slow Tokenization:**  
If you experience extremely slow tokenization, set `use_fast_tokenizer: false` in the LLaMA-Factory config files. See [issue #8600](https://github.com/hiyouga/LLaMA-Factory/issues/8600) for details.

**Slow Training with PyTorch 2.9:**  
Training may become extremely slow when using PyTorch 2.9 with Qwen3-VL. Consider downgrading to an earlier PyTorch version. See [Qwen3-VL issue #1701](https://github.com/QwenLM/Qwen3-VL/issues/1701) for more information.


## RA-HMD Stage 2 Code

Stage 1 fine-tuning delivers strong in-domain supervised performance. As shown in our ablation studies, Stage 2 does not significantly improve in-domain metrics. However, Stage 2 introduces a contrastive learning objective that enhances retrieval quality, leading to better generalization in cross-dataset, out-of-domain, and adversarial scenarios. Thus, if your focus is only on in-domain performance, Stage 1 suffices. The stage 2's complexity is not justified for marginal in-domain gains.

Stage 2 requires an additional preprocessing step:
- First, generate LMM embeddings using `RA-HMD/LLAMA-FACTORY/src/llamafactory/custom/extract_features.py`.
- Then, use the generated embeddings to refine the MLP layers via Stage 2 fine-tuning. The training scripts for Stage 2 are located in `RA-HMD/Stage2`.



# Citation

If you find our work useful for your research, please consider citing:
```
@inproceedings{RGCL2024Mei,
    title = "Improving Hateful Meme Detection through Retrieval-Guided Contrastive Learning",
    author = "Mei, Jingbiao  and
      Chen, Jinghong  and
      Lin, Weizhe  and
      Byrne, Bill  and
      Tomalin, Marcus",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.291",
    doi = "10.18653/v1/2024.acl-long.291",
    pages = "5333--5347"
}

@inproceedings{RAHMD2025Mei,
    title = "Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection",
    author = "Mei, Jingbiao  and
      Chen, Jinghong  and
      Yang, Guangyu  and
      Lin, Weizhe  and
      Byrne, Bill",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1215/",
    pages = "23817--23839",
    ISBN = "979-8-89176-332-6",
}



```
