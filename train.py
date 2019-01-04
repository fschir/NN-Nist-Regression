import tensorflow as tf
import os

import hy_param
import model

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)


#Feeds image
X = model.X
#Feeds label
Y = model.Y

checkpoint_dir = os.path.abspath(os.path.join(hy_param.checkpoint_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for step in range(1, hy_param.num_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(hy_param.batch_size)
        sess.run(model.train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % hy_param.display_step == 0 or step == 1:
            loss, acc = sess.run([model.loss_op, model.accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step: " + str(step) + ", Minibatch loss: " + "{:-4f}".format(loss) +
                  ", Training Acc: " + "{:.3f}".format(acc))
        if step % hy_param.checkpoint_every == 0:
            path = saver.save(
                sess, checkpoint_prefix, global_step=step)
            # print("Saved model checkpoint to: {}\n".format(path))

    print("Operation complete!\n")
    print("Testing Accuracy:  {:.4f}". format(sess.run(model.accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})))

