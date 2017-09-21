import numpy as np
import tensorflow as tf
import pandas as pd

tf.logging.set_verbosity(tf.logging.INFO)


#Takes data in the form of .csv files
#Transposes the data to make feature columns
#Adds labels to the data accoriding to cancer group
#Returns n shuffled rows (samples) where n is the batch size defined
def shuffle_data(batch, data):
    data1 = data.copy()
    samples = list(data1)
    GI_STR = samples[0]
    del samples[0]
    samples = np.random.permutation(samples)
    samples = samples.astype("U64")
    samples = np.insert(samples, 0, GI_STR)
    if samples[0][-1] != 'r':
        samples[0] = samples[0] + "r"

    data1 = data1[samples]

    #Use genes as features
    data1 = data1.set_index('GenomicIdentifier').T


    '''

    Cancer target column codes
    0 - BileDuct
    1 - BoneMarrow
    2 - Brain
    3 - Breast
    4 - Cervix
    5 - Pancreas
    6 - Pleura
    7 - Prostate
    8 - Skin
    9 - SoftTissue
    10 - Stomach
    
    '''

    #Cancers starting with 's', 'c', or 'v' had first
    #character cut off when preprocessing the data
    cancer_list = []
    for cancer in samples:
        if cancer[0:3].lower() == 'bil':
            cancer_list.append(0)
        elif cancer[0:3].lower() == 'bon':
            cancer_list.append(1)
        elif cancer[0:3].lower() == 'bra':
            cancer_list.append(2)
        elif cancer[0:3].lower() == 'bre':
            cancer_list.append(3)
        elif cancer[0:3].lower() == 'erv':
            cancer_list.append(4)
        elif cancer[0:3].lower() == 'pan':
            cancer_list.append(5)
        elif cancer[0:3].lower() == 'ple':
            cancer_list.append(6)
        elif cancer[0:3].lower() == 'pro':
            cancer_list.append(7)
        elif cancer[0:3].lower() == 'kin':
            cancer_list.append(8)
        elif cancer[0:3].lower() == 'oft':
            cancer_list.append(9)
        elif cancer[0:3].lower() == 'tom':
            cancer_list.append(10)
    

    CANCER_TARGET = np.array(cancer_list)

    training_data = data1.as_matrix()
    training_data = [x for x in training_data]
    training_targets = [x for x in CANCER_TARGET]

    
    return np.array(training_data[:batch], dtype='float32'), np.array(training_targets[:batch], dtype='float32'), np.array(samples[0:231])


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  input_layer = tf.reshape(features["x"], [-1,60483,1])

  # Convolutional Layer #1
  # Computes 8 features using a 100 x 1 filter with ReLU activation.
  # Input Tensor Shape: [batch_size, 60483, 1, 1]
  # Output Tensor Shape: [batch_size, 60483, 1, 8]
  conv1 = tf.layers.conv1d(
      inputs=input_layer,
      filters=8,
      kernel_size=[100],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 3x1 filter and stride of 3
  # Input Tensor Shape: [batch_size, 60483, 1, 8]
  # Output Tensor Shape: [batch_size, 20161, 1, 16]
  pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=[3], strides=3)

  # Convolutional Layer #2
  # Computes 16 features using a 100 x 1 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 20161, 1, 16]
  # Output Tensor Shape: [batch_size, 20161, 1, 32]
  conv2 = tf.layers.conv1d(
      inputs=pool1,
      filters=16,
      kernel_size=[100],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 34x1 filter and stride of 34
  # Input Tensor Shape: [batch_size, 20161, 1, 16]
  # Output Tensor Shape: [batch_size, 593, 1, 16]
  pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=[34], strides=34, padding='same')


  # Input Tensor Shape: [batch_size, 593, 1, 16]
  # Output Tensor Shape: [batch_size, 593 * 1 * 16]
  pool2_flat = tf.reshape(pool2, [-1, 593 * 1 * 16])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 593 * 1 * 16]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.8 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 11]
  logits = tf.layers.dense(inputs=dropout, units=11)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=11)
  
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
    optimizer = tf.train.AdamOptimizer(0.00003)
    #optimizer = tf.train.AdagradOptimizer(1e-2)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  data = pd.read_csv('D:\\cancer_research\\all_data\\training_data.csv', sep=',', header=0)
  train_data, train_labels, _ = shuffle_data(453, data)
  
  test = pd.read_csv('D:\\cancer_research\\all_data\\testing_data.csv', sep=',', header=0)
  eval_data, eval_labels, _ = shuffle_data(448, test)
  
  # Create the Estimator
  cancer_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="D:/tmp/genomic_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=453,
      num_epochs=None,
      shuffle=True)
  cancer_classifier.train(
      input_fn=train_input_fn,
      steps=500,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = cancer_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
