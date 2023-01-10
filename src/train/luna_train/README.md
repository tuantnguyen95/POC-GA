# Train Luna

Crawl train/test data from S3 and train 2 sets of weights:

**Weight 1 and 2 for features: OCR threshold 50, image resized no padding**

**Weight 1 and 2 for features: OCR threshold 0, image resized with padding**

**Weight 1: decide the number of elements in final vector**

**Weight 2: decide the importance of each element in final vector**

# Performance
| Accuracy | Luna Test V3 | Regression Test |
|----------|--------------|-----------------|
| Luna     | 95.54%       | 94.77%          |
# Configuration

Edit file [s3.json] for these configurations
`aws_access_key_id`
`aws_secret_access_key`

Params args:

**compare-weight-version**: weight version that we want to make a comparison with, default -1 mean get the max version on S3

**weight-version**: the training weight version, default -1 will define max+1

**data-version-min**, **data-version-max**: data versions that we want to train the weight from, default 0 -> -1 mean all data

**weight-init**: init the weights from this weight version, default -1 will train from scratch

# Requirement

* Python >= 3.7.3
* requirements.txt
  ``` pip3 install -r requirements.txt```

# Quickstart

```
python3 -m train --epochs 50 --batch-size 512 --weight-path weights/ --compare-weight-version 1 --weight-version 1 --weight-init -1 --data-path data/ --data-version-min 2 --data-version-max 3 --config-file s3.json
```