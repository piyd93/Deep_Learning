import tensorflow as tf
import tflib as lib
import tflib.linear
import tflib.conv1d
import __init__

#y = tf.placeholder(tf.float32, shape=[None, 128])


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def ResBlock(name, inputs, dim):
    # print("- Creating ResBlock -")
    output = inputs
    output = tf.nn.relu(output)
    output = lib.conv1d.Conv1D(name+'.1', dim, dim, 5, output)
    # print("After conv:", output)
    output = tf.nn.relu(output)
    output = lib.conv1d.Conv1D(name+'.2', dim, dim, 5, output)
    return inputs + (0.3*output)

def Generator(n_samples, seq_len, layer_dim, output_dim, prev_outputs=None):
    # print("- Creating Generator -")
    output = make_noise(shape=[n_samples, 128])
    # print("Initialized:", output)
    output = lib.linear.Linear('Generator.Input', 128, seq_len * layer_dim, output)
    # print("Lineared:", output)
    output = tf.reshape(output, [-1, seq_len, layer_dim,])
    # print("Reshaped:", output)
    output = ResBlock('Generator.1', output, layer_dim)
    output = ResBlock('Generator.2', output, layer_dim)
    output = ResBlock('Generator.3', output, layer_dim)
    output = ResBlock('Generator.4', output, layer_dim)
    output = ResBlock('Generator.5', output, layer_dim)
    output = lib.conv1d.Conv1D('Generator.Output', layer_dim, output_dim, 1, output)
    output = softmax(output, output_dim)
    return output

def Discriminator(inputs, seq_len, layer_dim, input_dim):
    output = inputs
    output = lib.conv1d.Conv1D('Discriminator.Input', input_dim, layer_dim, 1, output)
    output = ResBlock('Discriminator.1', output, layer_dim)
    output = ResBlock('Discriminator.2', output, layer_dim)
    output = ResBlock('Discriminator.3', output, layer_dim)
    output = ResBlock('Discriminator.4', output, layer_dim)
    output = ResBlock('Discriminator.5', output, layer_dim)
    output = tf.reshape(output, [-1, seq_len * layer_dim])
    output = lib.linear.Linear('Discriminator.Output', seq_len * layer_dim, 1, output)
    return output

def softmax(logits, num_classes):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, num_classes])
        ),
        tf.shape(logits)
    )

def make_noise(shape):
    return tf.random_normal(shape)


def sigmoid_cross_entropy_with_logits(x, y):
     try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
     except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)