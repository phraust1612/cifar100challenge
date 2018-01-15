import tensorflow as tf
import numpy as np

class Net:
  def __init__ (self):
    pass

  def load (self):
    """
    load ():
      load weight parameters from param/
    """
    for name in self.namelist:
      nptmp = np.load (self.param_dir+name)
      self.W[name] = tf.Variable (tf.convert_to_tensor(nptmp, name=name))

  def save (self, sess):
    """
    save (sess):
      save weight parameters
      <arguments>
        sess : tensorflow session
    """
    for name in self.namelist:
      nptmp = sess.run (self.W[name])
      np.save (self.param_dir+name, nptmp)

  def get_class (self, sess, image:np.ndarray):
    """
    get_class (sess, image):
      guess the class of input image
      <arguments>
        sess : tensorflow session
        image : numpy array of shape : (224, 224, 1)
    """
    if image.ndim < 3 or image.ndim > 3:
      return -1
    if image.shape != (224,224,3):
      return -1

    _feed = {self.x:self.resize_image (image), self.tf_drop:1}
    output = sess.run (self.output, feed_dict=_feed)
    return tf.argmax (output, 1)

  def get_output (self, sess, image):
    """
    get_output (sess, image):
      apply net and take the score function
      <arguments>
        sess : tensorflow session
        image : numpy array of shape : (224, 224, 1)
    """
    _feed = {self.x:self.resize_image (image), self.tf_drop:1}
    return sess.run (self.output, feed_dict=_feed)

  def get_accuracy (self, sess, feed):
    """
    get_output (sess, image):
      test net and take the accuracy
      <arguments>
        sess : tensorflow session
        feed : dict {'x', 'y'}
    """
    _feed = {self.x:self.resize_image (feed['x']), self.y:feed['y'], self.tf_drop:1}
    return sess.run (self.accuracy, feed_dict=_feed)

  def train_param (self, sess, feed):
    """
    train_param (sess, feed):
      train and return loss
      <arguments>
        sess : tensorflow session
        feed : dict {'x', 'y', 'drop'}
    """
    _feed = {self.x:self.resize_image (feed['x']), self.y:feed['y'], self.tf_drop:feed['drop']}
    c,_ = sess.run([self.loss, self.train], feed_dict=_feed)
    return c

  def resize_image (self, image:np.ndarray):
    """
    resize_image (image):
      resize image to 224 x 224 x 3
      <arguments>
        image : numpy array
    """
    if image.ndim < 3 or image.ndim > 3:
      return -1
    elif image.ndim == 3:
      width = 0
      height = 1
    else:
      width = 1
      height = 2

    if image.shape[width] < 224:
      image = np.repeat (image, (224 // image.shape[width]), axis=width)
    if image.shape[height] < 224:
      image = np.repeat (image, (224 // image.shape[height]), axis=height)

    return image
