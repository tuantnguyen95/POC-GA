# Multimodal training based on similarities

### Prerequisite
- Sklearn
- Tensorflow 2.x
- Python3


### Data

Data is stored on the GPU AWS instance:
```bash
ubuntu: ls /home/ubuntu/workspace/data/sklearn_data
screenshots  test.csv  train.csv  xmls
```


### Run the scripts

```bash
python multimodal_train.py PATH_TO_train.csv PATH_TO_train.test PATH_TO_to_sklearn_data PATH_TO_visual_model
```


|Model     | Precision     | Recall        | F1    |
|:----------| :----------: |:-------------:| :-----:|
|With OCR  | 0.98          | 0.96          | 0.97  |
|No OCR    | 0.97          | 0.94          | 0.96 |
Table: Results obtained on April 24th, 20.


### Notes

There are plenty of room for improvements in this script, I just quickly name few of them here:
- Handle missing values: Ex. an element does not have ocr text, text or id.
- The composition similarity value: Client need a number indicating how similar 2 elements are. For now, perceptron coefficient values are taken to be weights for each features in the composition function. There must be a better way to do so.
- The script is really straightforward. This is because it needed to be done in a quite strict time limit. However, members of the teamm has been testing many different approaches for this classifier, inlcuding: triplet loss, graph, RNNs ... But none of them can surpass this simple classifier. Not even close!!! We doubt that lack of training data is the cause. Please try to improve this again once we have obtain a good amount of data. Right now we have less than 10k positives.



