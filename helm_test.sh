# Run benchmark
helm-run --conf-paths federatedscope/llm/eval/eval_for_helm/run_specs.conf --enable-local-huggingface-model decapoda-research/llama-7b-hf --suite test -m 100 --local -n 1 --yaml federatedscope/llm/baseline/llama.yaml --ckpt_dir xxxx --skip-completed-runs --local-path xxx

# # Summarize benchmark results
# helm-summarize --suite v1

# # Start a web server to display benchmark results
# helm-server

# # Remove the configuration file 
# rm run_specs.conf