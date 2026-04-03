export NAME=LLM-JEPA-1B

python finetune.py \
  --train_file datasets/synth_train.jsonl \
  --output_dir $NAME \
  --model_name HuggingFaceTB/SmolLM2-135M-Instruct \
  --learning_rate 2e-5 \
  --num_epochs 6 \
  --predictors 1 \
  --batch_size 1 \
  --grad_accum 16 \
  --lbd 1.0