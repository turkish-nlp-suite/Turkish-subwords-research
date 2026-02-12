# Running benchmarking on BERT models

This folder has 3 scripts, the main script includes TrGLUE running is `run.py`. This script will pull TrGLUE datasets from [TrGLUE HF repo](https://huggingface.co/datasets/turkish-nlp-suite/TrGLUE).

For running all the GLUE tasks on all the models simply run `./run_all_tasks.sh` . It'll download all the BERT models from their repos on [HF](https://huggingface.co/collections/turkish-nlp-suite/turkish-subwords-research). This script iterates over all the vocab sizes and corpora sizes. 

If you wanna run a single task, we put an example on TrSST-2, please checkout `run_single.sh`, one can replace task names and params by looking at the `run_all_tasks.sh` script. 
