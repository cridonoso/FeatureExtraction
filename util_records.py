import tensorflow as tf 


def read_tfrecord(serialized_example):
    context_features = {
        "label": tf.io.FixedLenFeature([], dtype=tf.int64),
        "n_classes": tf.io.FixedLenFeature([], dtype=tf.int64)
    }
    
    sequence_features = {
        "mjd": tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        "mags": tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        "err_mags": tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        "mask": tf.io.FixedLenSequenceFeature([], dtype=tf.float32)
    }

    context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        sequence_features=sequence_features,
        context_features =context_features
    )
    
    x = tf.stack([sequence_parsed['mjd'],
                  sequence_parsed['mags'],
                  sequence_parsed['err_mags']
                  ], axis=1)
    
    y = context_parsed['label']
    #y_one_hot = tf.one_hot(y, tf.cast(context_parsed['n_classes'], tf.int32))
    m = sequence_parsed['mask']
    
    return x, y, m

def load_record(path, batch_size, n_samples=-1):
    """ Data loader for irregular time series with masking"
    
    Arguments:
        path {[str]} -- [record location]
        batch_size {[number]} -- [number of samples to be used in 
                                  neural forward pass]
        n_samples {[number]} -- [Number of samples to use. 
                                 By default all elements are returned (-1)]
    Returns:
        [tensorflow dataset] -- [batches to feed the model]
    """
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(lambda x: read_tfrecord(x))
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache
    if n_samples != -1:
        dataset = dataset.take(n_samples)
    dataset = dataset.cache() 
    batches = dataset.batch(batch_size)
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch
    batches = batches.prefetch(buffer_size=1)
    return batches
