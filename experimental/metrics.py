
### Мера для кластеризации (bcb)
import random
import numpy as np

def random_segmentation(true_seg):
    claster_count = len(np.unique(true_seg))
    random_clasters = []
    for _ in true_seg:
        random_clasters.append(random.choice(range(claster_count)))
    return random_clasters
    
def bcb_all(element, pred_clusters, true_clusters):
    element_pred_cluster = []
    element_true_cluster = []
    for i in range(len(pred_clusters)):
        if pred_clusters[i] == pred_clusters[element]:
            element_pred_cluster.append(i)
    for i in range(len(true_clusters)):
        if true_clusters[i] == true_clusters[element]:
            element_true_cluster.append(i)
    true_intersection = []
    for elm in element_pred_cluster:
        if elm in element_true_cluster:
            true_intersection.append(elm)
    bcb_precision = len(true_intersection) / len(element_pred_cluster)
    bcb_recall = len(true_intersection) / len(element_true_cluster)
    bcb_f1 = 2*bcb_precision*bcb_recall / (bcb_precision + bcb_recall)
    return bcb_precision, bcb_recall, bcb_f1   

def bcb_clust(pred_clusters, true_clusters):
    bcb_p_avg = 0
    bcb_r_avg = 0
    bcb_f1_avg = 0
    for elm in range(len(pred_clusters)):
        bcb_p_elm, bcb_r_elm, bcb_f1_elm = bcb_all(elm, pred_clusters, true_clusters)
        bcb_p_avg += bcb_p_elm
        bcb_r_avg += bcb_r_elm
        bcb_f1_avg += bcb_f1_elm
    bcb_p_avg /= len(pred_clusters)
    bcb_r_avg /= len(pred_clusters)
    bcb_f1_avg /= len(pred_clusters)
    return bcb_p_avg, bcb_r_avg, bcb_f1_avg