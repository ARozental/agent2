
from recognizer.pair_generator import PairGenerator
from recognizer.tf_dataset import Dataset
from recognizer.model import Model
import tensorflow as tf
import pylab as plt
import numpy as np


def main():
    generator = PairGenerator()
    # print 2 outputs from our generator just to see that it works:
    iter = generator.get_next_pair()
    for i in range(2):
        print(next(iter))
    ds = Dataset(generator)
    model_input = ds.next_element
    model = Model(model_input)

    # train for 100 steps
    with tf.compat.v1.Session() as sess:
        # sanity test: plot out the first resized images and their label:
        (img1, img2, label) = sess.run([model_input.img1,
                                        model_input.img2,
                                        model_input.label])

        # img1 and img2 and label are BATCHES of images and labels. plot out the first one
        plt.subplot(2, 1, 1)
        plt.imshow(img1[0].astype(np.uint8))
        plt.subplot(2, 1, 2)
        plt.imshow(img2[0].astype(np.uint8))
        plt.show()

        # intialize the model
        sess.run(tf.compat.v1.global_variables_initializer())
        # run 100 optimization steps
        for step in range(100):
            (_, current_loss) = sess.run([model.opt_step,
                                          model.loss])


if __name__ == '__main__':
    main()
