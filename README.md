
This code is provided as a reference of the code used to run experiments in the paper.
Please use https://github.com/negrinho/deep_architect instead if you plan to build on our language,
as that is the repo that will be maintained going forward.

To run this code, go to the root directory and run `python main.py` with the
required arguments.

```
Usage: main.py [-h] [--search-space {genetic,nasnet,nasbench,flat}]
               [--searcher {random,mcts,smbo,evolution}] --data-dir DATA_DIR
               [--tpu-name TPU_NAME] [--use-tpu]
               [--evaluation-dir EVALUATION_DIR] [--num-samples NUM_SAMPLES]
```

If you are training locally not on a TPU, then you can ignore the next
paragraph.

If using a TPU for training, the `--tpu-name` and `--use-tpu` parameters are
required. Furthermore, the `--data-dir` and `--evaluation-dir` arguments must
be directories in Google Cloud Storage, and you must have the
`GOOGLE_APPLICATION_CREDENTIALS` environment variable set to the file
containing your Google service key (see
https://cloud.google.com/docs/authentication/getting-started).

The `--search-space` argument takes in the name of the search space for the
search. The values `genetic`, `nasnet`, `nasbench`, and `flat` are supported.

The `--searcher` argument takes in the name of the searcher for the search.
The values `random`, `mcts`, `smbo`, and `evolution` are supported.

The `--data-dir` argument takes in the name of the directory where CIFAR-10
TFRecords are. Run `python datasets/generate_cifar10_tfrecords.py` to generate
the files. Required argument.

The `--evaluation-dir` argument takes in the name of the directory where the
Tensorflow estimator will produce checkpoint and summary files.

The `--num-samples` argument takes in how many architectures you want to sample
during the search.
