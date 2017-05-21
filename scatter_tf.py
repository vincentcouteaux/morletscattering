"""! computes the scatter transform of 2D images with tensorflow, enabling gpu
usage"""
import tensorflow as tf
import numpy as np
from morlet import morlet_bank

class Scatterer:
    """ instantiate this class with a list of wavelets and a low-pass
    filter, and scatter images """
    def __init__(self, phi, psis, sub_factor=8):
        """ phi is the low pass filter, psis a list of wavelets """
        psis = np.stack(psis, 2)
        psis = np.expand_dims(psis, 2)
        self.n_psis = psis.shape[3]
        self._re_psis = tf.constant(np.real(psis), dtype=tf.float32)
        self._im_psis = tf.constant(np.imag(psis), dtype=tf.float32)
        phi = np.reshape(phi, (phi.shape[0], phi.shape[1], 1, 1))
        self.phi = tf.constant(np.repeat(phi, psis.shape[3], 2), dtype=tf.float32)
        self.sub_factor = sub_factor
        self._build_graph(phi)
        self._sess = tf.Session()

    def _build_graph(self, phi):
        """ Create the tensors useful for computing 1st and 2nd scattering
        coefficients """
        sub_factor = self.sub_factor
        self.batch_ph = tf.placeholder(tf.float32)
        self.wvlt1_re = tf.nn.conv2d(self.batch_ph, self._re_psis,
                                     strides=[1, 1, 1, 1], padding="SAME")
        self.wvlt1_im = tf.nn.conv2d(self.batch_ph, self._im_psis,
                                     strides=[1, 1, 1, 1], padding="SAME")
        self.wvlt1_mod = tf.sqrt(self.wvlt1_re**2 + self.wvlt1_im**2)
        self.order1 = tf.nn.depthwise_conv2d(self.wvlt1_mod, self.phi,
                                             strides=[1, 1, 1, 1],
                                             padding="SAME")
        self.out1 = tf.nn.avg_pool(self.order1, ksize=[1, sub_factor, sub_factor, 1],
                                   strides=[1, sub_factor, sub_factor, 1],
                                   padding="SAME")
        # channels = [] # list of tensors containing the channels
        channels = tf.split(self.wvlt1_mod, self.n_psis, axis=3)
        self.wvlt2_re_l = []
        self.wvlt2_im_l = []
        for channel in channels:
            self.wvlt2_re_l.append(tf.nn.conv2d(channel, self._re_psis,
                                                strides=[1, 1, 1, 1],
                                                padding="SAME"))
            self.wvlt2_im_l.append(tf.nn.conv2d(channel, self._im_psis,
                                                strides=[1, 1, 1, 1],
                                                padding="SAME"))
        self.wvlt2_re = tf.concat(self.wvlt2_re_l, axis=3)
        self.wvlt2_im = tf.concat(self.wvlt2_im_l, axis=3)
        self.wvlt2_mod = tf.sqrt(self.wvlt2_re**2 + self.wvlt2_im**2)
        self.phi2 = tf.constant(np.repeat(phi, self.n_psis**2, 2),
                                dtype=tf.float32)
        self.order2 = tf.nn.depthwise_conv2d(self.wvlt2_mod, self.phi2,
                                             strides=[1, 1, 1, 1],
                                             padding="SAME")
        self.out2 = tf.nn.avg_pool(self.order2, ksize=[1, sub_factor, sub_factor, 1],
                                   strides=[1, sub_factor, sub_factor, 1],
                                   padding="SAME")

    def scatter_batch(self, batch):
        """ feed a batch to the scattering network. batch has shape
        (batch, height, width, 1) """
        return self._sess.run((self.out1, self.out2), feed_dict={self.batch_ph: batch})

if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    from display import big_image
    #pylint: disable=C0103
    lena = np.array(Image.open("lena512.bmp"))
    #lena = lena[240:256, 240:256]
    #plt.imshow(lena)
    #plt.show()
    print("generating wavelet bank")
    phi, psis, _ = morlet_bank(16, js=[3, 4, 5])
    print("initializing graph")
    scat = Scatterer(phi, psis, 8)
    lena = np.reshape(lena, (1, lena.shape[0], lena.shape[1], 1))
    print("beginning scattering...")
    coefs = scat.scatter_batch(lena)
    print("end scattering")
    print(coefs[0].shape, coefs[1].shape)
    order2 = np.transpose(np.squeeze(coefs[1]), (2, 0, 1))
    plt.imshow(big_image(order2))
    plt.show()
