import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy
def wordEmbeddings():
        elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        embeddings = elmo(
        ["the cat is on the mat and i can't understand why", "dogs are in the fog"],
        signature="default",
        as_dict=True)["elmo"]
        print (embeddings.shape)
        return embeddings
wordEmbeddings()