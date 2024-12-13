
# For multiple seeds (save as run_seeds.sh):
#!/bin/bash
MODEL="/Users/screen-club/Documents/projets/liminal/motion-diffusion-model-mps-hack/pretrained_models/humanml_enc_512_50steps/model000750000.pt"
PROMPT="a person walking in circles fast"

for seed in 1 2 3 4 5; do
    python -m sample.generate \
        --model_path "$MODEL" \
        --num_samples 1 \
        --num_repetitions 1 \
        --text_prompt "$PROMPT" \
        --seed $seed
done