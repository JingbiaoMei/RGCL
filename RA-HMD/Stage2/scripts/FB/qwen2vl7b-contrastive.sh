feat_path="data/Embedding/qwen2vl/qwen2vl7b/FB/train_FB_2025-01-01_baseline-clssifier_checkpoint-950.pt"
cls_path=".../LLaMA-Factory/checkpoints/stage1/qwen2_vl-7b/hm/2025-01-01_baseline-clssifier/checkpoint-950"
# Note that feature_path is the path to the features extracted by the LMM backbone
# CLS_path is the path to the classifier model checkpoint
# Here, only the CLS model is trained based on the extracted feature to save time and GPU memory.

python ./src/run_rac_lmm.py --batch_size 64 --force \
    --lr 0.0001  --epochs 30  --topk 20 --dataset "FB" \
    --feature_path "$feat_path" \
    --cls_path "$cls_path" \
    --hard_negatives_loss --no_hard_negatives 1 --in_batch_loss \
    --seed 1 \
    --metric "ip" --loss "contrastive" \
    --hybrid_loss \
    --majority_voting "arithmetic" \
    --no_pseudo_gold_positives 1 \
    --Faiss_GPU --grad_clip 1.0 --weight_decay 0.01 