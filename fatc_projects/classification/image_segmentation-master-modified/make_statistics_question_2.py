import matplotlib.pyplot as plt
import pickle
import numpy as np

all_accuracies = pickle.load(open('result_pickles/accuracies_all.pickle', 'rb'))
bayes_rgb = np.array([a['accuracies_bayes_rgb'] for a in all_accuracies]).ravel()
bayes_shape = np.array([a['accuracies_bayes_shape'] for a in all_accuracies]).ravel()
knn_rgb = np.array([a['accuracies_knn_rgb'] for a in all_accuracies]).ravel()
knn_shape = np.array([a['accuracies_knn_shape'] for a in all_accuracies]).ravel()
majority = np.array([a['accuracies_majority'] for a in all_accuracies]).ravel()


def create_IC_interval(np_array):
    t_val = 1.96
    len_array = len(np_array)
    mean_array = np.mean(np_array)
    std_array = np.std(np_array, ddof=1)
    int_lower_array = mean_array - t_val*std_array/np.sqrt(len_array)
    int_high_array = mean_array + t_val*std_array/np.sqrt(len_array)
    return [mean_array, std_array, int_lower_array, int_high_array]

print 'BAYES-----------'

ic_bayes_rgb = create_IC_interval(bayes_rgb)
print 'RGB'
print 'Mean', ic_bayes_rgb[0]
print 'Std', ic_bayes_rgb[1]
print 'IC 95 [', ic_bayes_rgb[2], ',' , ic_bayes_rgb[3], ']'

ic_bayes_shape = create_IC_interval(bayes_shape)
print 'SHAPE'
print 'Mean', ic_bayes_shape[0]
print 'Std', ic_bayes_shape[1]
print 'IC 95 [', ic_bayes_shape[2], ',' , ic_bayes_shape[3], ']'

print 'KNN-----------'

ic_knn_rgb = create_IC_interval(knn_rgb)
print 'RGB'
print 'Mean', ic_knn_rgb[0]
print 'Std', ic_knn_rgb[1]
print 'IC 95 [', ic_knn_rgb[2], ',' , ic_knn_rgb[3], ']'

ic_knn_shape = create_IC_interval(knn_shape)
print 'SHAPE'
print 'Mean', ic_knn_shape[0]
print 'Std', ic_knn_shape[1]
print 'IC 95 [', ic_knn_shape[2], ',' , ic_knn_shape[3], ']'

print 'MAJORITY-----------'

ic_majority = create_IC_interval(majority)
print 'Mean', ic_majority[0]
print 'Std', ic_majority[1]
print 'IC 95 [', ic_majority[2], ',' , ic_majority[3], ']'


print 'Saving histograms'

plt.xlabel('acuracia BAYES RGB')
plt.ylabel('frequencia amostral')
plt.hist(bayes_rgb)
plt.savefig('result_images/bayes_accuracies_rgb.png')
plt.gcf().clear()

plt.xlabel('acuracia BAYES SHAPE')
plt.ylabel('frequencia amostral')
plt.hist(bayes_shape)
plt.savefig('result_images/bayes_accuracies_shape.png')
plt.gcf().clear()

plt.xlabel('acuracia KNN RGB')
plt.ylabel('frequencia amostral')
plt.hist(knn_rgb)
plt.savefig('result_images/knn_accuracies_rgb.png')
plt.gcf().clear()

plt.xlabel('acuracia KNN SHAPE')
plt.ylabel('frequencia amostral')
plt.hist(knn_shape)
plt.savefig('result_images/knn_accuracies_shape.png')
plt.gcf().clear()

plt.xlabel('acuracia Voto Majoritario')
plt.ylabel('frequencia amostral')
plt.hist(majority)
plt.savefig('result_images/majority_vote.png')
plt.gcf().clear()

print 'Done'
