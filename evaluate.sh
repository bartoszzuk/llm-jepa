python evaluate.py --model_name out/LLM-JEPA-135M \
  --input_file datasets/synth_test.jsonl \
  --original_model_name HuggingFaceTB/SmolLM2-135M-Instruct \
  --nosplit_data \
  --split_tune_untune \
  --output_file eval \
   --max_new_tokens 128