This is subdir for running word-level vocab NER experiment.

Rin the experiment directly by navigating to `training/` and do a `./run.sh`. To run the experiment with fractions, one need vocab file corresponding to that fractions. Vocab files are located under `training/vocab_files/` . The vocab file is passed to the script with `$vocab_file` param in the running script.

If you wanna generate vocab files from scratch, navigate to `make_vocabs/`. First run `./run_fracs.sh` to generate coverage files, then run `./write_manifest.sh` to write vocab files under `training/vocab_files/`. 
