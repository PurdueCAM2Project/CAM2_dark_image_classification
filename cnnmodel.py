
# Imports
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


# Our application logic will be added here


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer

    input_layer = tf.reshape(features["x"], [-1, 40, 30, 3])
    print(input_layer)
    input_layer = tf.cast(input_layer, dtype=tf.float32)
    input_norm = tf.layers.batch_normalization(input_layer)
    print(input_norm)
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_norm,
        filters=16,
        kernel_size=[5, 5],
        padding='valid',
        activation=tf.nn.relu)
    print(conv1)

    # Pooling Layer #1
    # conv1_norm = tf.layers.batch_normalization(conv1)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)
    pool1 = tf.layers.batch_normalization(pool1)
    print(pool1)

    # # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[5, 5],
        padding="valid",
        activation=tf.nn.relu)
    print(conv2)
    # conv2 = tf.layers.batch_normalization(conv2)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2 = tf.layers.batch_normalization(pool2)
    print(pool2)

    pool2_flat = tf.reshape(pool2, [-1, 4*6*32])
    # dconv1=tf.layers.conv2d_transpose()
    # Dense Layer
    dense1 = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu)  # 36/2
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense2 = tf.layers.dense(inputs=dropout1, units=80, activation=tf.nn.relu)
    dense3 = tf.layers.dense(inputs=dense2, units=2, activation=tf.nn.relu)
    # dropout = tf.layers.dropout(
    #     inputs=dense3, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    output = dense3

    predictions = {
        'val': output
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=output)
    # loss = tf.losses.mean_squared_error(
    #     labels=labels, predictions=output)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        rate = tf.train.exponential_decay(learning_rate=0.1, global_step=tf.train.get_global_step(), decay_steps=100,
                                          decay_rate=0.9, staircase=False)
        train_op = tf.train.AdadeltaOptimizer(learning_rate=rate).minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        celoss= tf.losses.softmax_cross_entropy(onehot_labels=labels,logits=output)
        eval_metric_ops = { "error": celoss}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss)

def main(unused_argv):
    # Load training and eval data
    datax = np.load('x.npy')
    datay = np.load('y.npy')
    # datax = datax.reshape([-1, 40, 40, 6])
    # my_feature_columns = [tf.feature_column.numeric_column(key='x', shape=[14400])]
    train_data = datax[0:340]  # Returns np.array
    train_labels = datay[0:340]

    eval_data = datax[300:340]
    eval_labels = datay[300:340]

    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn
    )

    # Set up logging for predictions
    tensors_to_log = {"val"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=128,
        num_epochs=100,
        shuffle=True,
        num_threads=8)
    train_res = estimator.train(
        input_fn=train_input_fn,

    )
    print('train result')
    print(train_res)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = estimator.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    # x = estimator.predict(input_fn=eval_input_fn)


tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

if __name__ == "__main__":
    tf.app.run()
