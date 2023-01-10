# Facenet Tensorflow 2.1
These codes are referred at [David Sandberg's repo](https://github.com/davidsandberg/facenet/tree/master/src)

## Prerequisites
* CUDA 10.0 or 10.1
* Tensorflow 2.0++
* tf_slim at the [commit](https://github.com/google-research/tf-slim/commit/b62d6d8698243a1737ceeb5a929a435c06a577ae)
## Training
* *Note*: The best way to train is go to the EC2 GPU machine folder `ec2-52-74-220-169.ap-southeast-1.compute.amazonaws.co:/home/duc/facenet/src` because it requires GPU pc and configured prerequisites for trainning.

Run command:
```shell script
python train_softmax_v2.py \
--logs_base_dir ~/logs/facenet_v2/ \
--models_base_dir ~/models/facenet_v2/ \
--data_dir ../output/size_260/ \
--image_size 260 \
--model_def models.inception_resnet_v2_v2 \
--optimizer ADAM \
--learning_rate -1 \
--max_nrof_epochs 150 \
--keep_probability 0.8 \
--use_fixed_image_standardization \
--learning_rate_schedule_file ../data/learning_rate_schedule_classifier_casia.txt \
--weight_decay 5e-4 \
--embedding_size 128 \
--validation_set_split_ratio 0.4 \
--validate_every_n_epochs 5 \
--epoch_size 500 \
--batch_size 16
```

## Result
### Baseline model:
* Model: Inception Resnet v2
* Accuracy: 0.942
* Epoch: 50
* Keep prob: 0.8
* learning rate

| Epoch | Value |
| ----- | ----- |
|   0   |  0.1  |
|   25  |  0.05 |
|   35  |  0.01 |
|   45  |  0.05 |
|   50  |  0.001|

#### Fine-tuning:

| No | keep_prob | Epoch | Augment data | Accuracy |
| -- | --------- | ----- | ------------ | -------- |
| 1  |    0.7    |   50  |      NO      |   0.943  |
| 2  |    0.5    |   90  |      YES     |   0.942  | New lr schedule, bright, reach 0.941 at 50 epochs |
| 3  |    0.5    |   50  |      YES     |   N/A    |                     Saturate                      |
