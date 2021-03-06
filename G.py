"""
Simon Shenmeng Wang
simon.wang.orz@gmail.com
ssw8641@nyu.edu
"""

import numpy as np

class G():

    def __init__(self, model, model_fixed):
        # super(G, self).__init__()
        self.model = model
        self.model_fixed = model_fixed

    def predict(self, x):
        final_out = []
        output = self.model.predict(x)
        output_fixed = self.model.predict(x)
        cls = output.shape[1]
        for i in range(len(output)):
            if np.argmax(output[i]) != np.argmax(output_fixed[i]):
                output_G = np.zeros(cls + 1)
                output_G[cls] = 1
            else:
                output_G = np.append(output[i], 0)
            final_out.append(output_G)
        return np.array(final_out)
