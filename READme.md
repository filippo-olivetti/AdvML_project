---
runme:
  id: 01HK7Z8CZ0ZYBBGG6CB9HD29QC
  version: v2.0
---

## Goal

The goal of this repository is to build and train a basic character-level Recurrent Neural Network (RNN) to classify words. For efficiency reasons we will use LSTM network (which is a variant of RNN network).

A character-level RNN reads words as a series of characters - outputting a prediction and “hidden state” at each step, feeding its previous hidden state into each next step. We take the final prediction to be the output, i.e. which class the word belongs to.

Specifically, we’ll train on a few thousand surnames from 18 languages of origin, and predict which language a name is from based on the spelling.

```sh {"id":"01HK7ZDXE5BGAJM3F3RQQVPYDX"}
$ python predict.py Hinton
(-0.47) Scottish
(-1.52) English
(-3.57) Irish

$ python predict.py Schmidhuber
(-0.19) German
(-2.48) Czech
(-2.68) Dutch
```

Results are strongly dependent on the dataset composition, indeed the dataset is divided into

Data statistics: 
English 3668
German 724
Arabic 2000
Greek 203
Korean 94
Italian 709
Irish 232
Dutch 297
Vietnamese 73
Portuguese 74
Czech 519
Japanese 991
Polish 139
Spanish 298
Russian 9408
Chinese 268
Scottish 100
French 277

Clearly the network have more difficulties in correctly classifying Vietnamese and Portuguese names, and is very good in Italian, Russian, German names. It is interesting to see that many English surnames are misclassified as Scottish and Irish. The following are the results after 15 epochs of training (overall accuracy is 76%).

```sh {"id":"01HK8HRFAWQKAMNB7EQ4WY8F25"}
              precision    recall  f1-score   support

           0       0.86      0.93      0.89       941
           1       0.44      0.36      0.39        73
           2       0.64      0.78      0.70       367
           3       0.00      0.00      0.00         8
           4       0.64      0.76      0.69       100
           5       0.67      0.27      0.38        30
           6       0.83      0.94      0.88       200
           7       0.33      0.12      0.17        52
           8       0.67      0.14      0.24        14
           9       0.65      0.62      0.63        21
          10       0.00      0.00      0.00        28
          11       0.00      0.00      0.00         8
          12       0.44      0.30      0.36        27
          13       0.00      0.00      0.00        10
          14       0.00      0.00      0.00        10
          15       1.00      0.07      0.12        30
          16       0.00      0.00      0.00        24
          17       0.62      0.65      0.63        71

    accuracy                           0.76      2014
   macro avg       0.43      0.33      0.34      2014
weighted avg       0.72      0.76      0.73      2014
```

Inspired by:

- https://github.com/claravania/lstm-pytorch/blob/master/train.py
- https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html