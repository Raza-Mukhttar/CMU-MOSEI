# -*- coding: utf-8 -*-
"""Dataset Alignments.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HcfMXvhqNJ75G6lqvXdaYBg9Yc5ooL8B

# **Dataset Alignments**

**Align glove_vectors**

Using average function for intervals and features to summarize the other modalities based on a same set of function

**Issues faced:** Due to weak internet connection, and low cmputation power of system, after some time system crashed when align the glove vectors with specifiec everage function of interval and features

Without fucntion the system also crashed as i tried in both ways
"""

#import numpy
#def myavg(intervals,features):
#        return numpy.average(features,axis=0)

dataset.align('glove_vectors') #,collapse_functions=[myavg])

"""**Add compututional squences for labels in the dataset**

Align all computational sequences according to the labels of a dataset. First, we fetch the opinion segment labels (All Labels) computational sequence for CMU-MOSEI

After aligning computational sequences align all the labels. Since every video has multiple segments according to annotations and timing label in each video
"""

dataset.add_computational_sequences(mmdatasdk.cmu_mosei.labels,'cmumosei/')
dataset.align('All Labels')