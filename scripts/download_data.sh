#!/usr/bin/env bash
# Raw inputs. All public; total ~600 MB.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
mkdir -p lesci datadecide epoch configs
HF=https://huggingface.co
for m in 70m 160m 410m 1.4b 2.8b 6.9b 12b; do   # Lesci et al. 2024 (no 1B split was released)
  curl -sL -o lesci/$m.parquet $HF/datasets/pietrolesci/pythia-deduped-stats/resolve/main/data/$m-00000-of-00001.parquet
done
curl -sL -o datadecide/ppl.parquet $HF/datasets/allenai/DataDecide-ppl-results/resolve/main/data/train-00000-of-00001.parquet
curl -sL -o epoch/svg_extracted_data.csv https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/main/data/svg_extracted_data.csv
for s in 4M 6M 8M 10M 14M 16M 20M 60M 90M 150M 300M 530M 750M 1B; do
  curl -sL -o configs/dd_$s.json $HF/allenai/DataDecide-dolma1_7-$s/resolve/main/config.json
done
for s in 70m 160m 410m 1b 1.4b 2.8b 6.9b 12b; do
  curl -sL -o configs/pythia_$s.json $HF/EleutherAI/pythia-$s-deduped/resolve/main/config.json
done
