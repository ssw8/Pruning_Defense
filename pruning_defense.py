import h5py
import numpy as np
import keras
import matplotlib.pyplot as plt

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data


# file names
clean_data_filename = '/data/clean_validation_data.h5'
poisoned_data_filename = '/data/sunglasses_poisoned_data.h5'
model_filename = '/models/bd_net.h5'


# data-loading
cl_x_test, cl_y_test = data_loader(clean_data_filename)
bd_x_test, bd_y_test = data_loader(poisoned_data_filename)
model = keras.models.load_model(model_filename)


# eval
def eval(model, cl_x_test=cl_x_test, cl_y_test=cl_y_test,
         bd_x_test=bd_x_test, bd_y_test=bd_y_test):
    cl_label_p = np.argmax(model.predict(cl_x_test), axis=1)
    clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test)) * 100
    print('Clean Classification accuracy:', clean_accuracy)

    bd_label_p = np.argmax(model.predict(bd_x_test), axis=1)
    asr = np.mean(np.equal(bd_label_p, bd_y_test)) * 100
    print('Attack Success Rate:', asr)

    return clean_accuracy, asr


print('Initial results: ')
init_clean_accuracy, init_asr = eval(model)

# obtaining feature indices in increasing order of pool_3 activations
model_clone = keras.models.clone_model(model)
model_clone.set_weights(model.get_weights())
cl_accuracy = []
asr = []
model_archive = []
features = []

pool3_out = model_clone.get_layer('pool_3').output
model_clone_truncated = keras.models.Model(inputs=model_clone.input,
                                           outputs=pool3_out)
model_clone_truncated_out = model_clone_truncated.predict(cl_x_test)
feature_results = np.mean(model_clone_truncated_out, axis=(0, 1, 2))
feature_rank = np.argsort(feature_results)


def prune(model, threshold, feature_rank=feature_rank,
          init_clean_accuracy=init_clean_accuracy):
    cl_accuracy = []
    asrs = []
    model_archive = []
    # curr_features = np.arange(len(feature_rank)).tolist()
    feature_rank = feature_rank.tolist()

    new_acc = 100
    model_pre = model
    i = 0

    while new_acc >= (init_clean_accuracy - threshold) and len(feature_rank) != 0:
        feature = feature_rank[0]
        feature_rank.remove(feature)
        model_curr = keras.models.clone_model(model_pre)
        model_curr.set_weights(model_pre.get_weights())
        conv3_weights = model_curr.get_layer('conv_3').get_weights()
        w = conv3_weights[0]
        b = conv3_weights[1]
        w[:, :, :, feature] = np.zeros([w.shape[0], w.shape[1], w.shape[2]])
        b[feature] = 0
        model_curr.get_layer('conv_3').set_weights([w, b])
        print('iter ', i)
        i += 1
        print('feature ', feature)
        new_acc, asr = eval(model_curr)
        cl_accuracy.append(new_acc)
        asrs.append(asr)

        model_pre = model_curr

    return [model_pre, cl_accuracy, asrs]

thresholds = [2, 4, 10, 30, 100]

acc_log_full = None
asr_log_full = None

for t in thresholds:
    new_model, acc_log, asr_log = prune(model, t)
    model_name = '/models/repaired_model_' + str(t) + '.h5'
    new_model.save(model_name)
    if t == 100:
        acc_log_full = acc_log
        asr_log_full = asr_log

N = pool3_out.shape[3]
x_axis = np.arange(len(acc_log_full))/N
plt.plot(x_axis, acc_log_full)
plt.plot(x_axis, asr_log_full)
plt.legend(['Accuracy Rate', 'Attack Success Rate'])
plt.xlabel('Fraction of Channels Pruned')
plt.ylabel('Accuracy / Attack Success Rate')
plt.savefig('acc_asr_vs_pruned_channel.jpg')




