"""
Simon Shenmeng Wang
simon.wang.orz@gmail.com
ssw8641@nyu.edu
"""

import keras
import numpy as np
import G
from pruning_defense import data_loader

clean_data_filename = '/data/clean_test_data.h5'
poisoned_data_filename = '/data/sunglasses_poisoned_data.h5'
bd_model_filename = '/models/bd_net.h5'

repaired_model_filename = 'models/repaired_model_30.h5'
"""
Alternatively, use 
'models/repaired_model_2.h5', or
'models/repaired_model_4.h5', or
'models/repaired_model_10.h5'
for repaired_model_filename
to experiment with different repaired models based on their 
fractions of pruned channels
"""



def main():
    cl_x_test, cl_y_test = data_loader(clean_data_filename)
    bd_x_test, bd_y_test = data_loader(poisoned_data_filename)

    bd_model = keras.models.load_model(bd_model_filename)
    repaired_model = keras.models.load_model(repaired_model_filename)

    goodnet = G(bd_model, repaired_model)

    cl_label_p = np.argmax(goodnet.predict(cl_x_test), axis=1)
    clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test)) * 100
    print('Clean Classification accuracy:', clean_accuracy)

    goodnet_result = goodnet.predict(bd_x_test)
    bd_label_p = goodnet_result[:, goodnet_result.shape[1] - 1]
    asr = np.mean(np.equal(bd_label_p, bd_y_test)) * 100
    print('Attack Success Rate:', asr)


if __name__ == '__main__':
    main()
