import os
import struct
import numpy as np

class NueralUtil:


    def load_mnist(self,path,kind='train'):

        """ Load MNIST data from defined path """
        labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
        images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))

            labels = np.fromfile(lbpath, dtype=np.uint8)

        with open(images_path, 'rb') as imagePath:

            magic, num, rows, cols = struct.unpack(">IIII", imagePath.read(16))

            images = np.fromfile(imagePath, dtype=np.uint8).reshape(len(labels),784)

            images = ((images/255.) - .5) * 2

        return images, labels