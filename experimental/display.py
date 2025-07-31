from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import numpy as np

def create_tsne_for_plot(data, labels, highlight = None, size = None, nc = 10, color_map = cm.cool ):
    np.random.seed(0)
    max_label = max(labels)
    # max_items = np.random.choice(range(data.shape[0]), size=len(labels), replace=False)
    
    # pca = PCA(random_state = 1, n_components=2).fit_transform(np.asarray(data))
    tsne = TSNE(random_state = 1, n_iter = 5000).fit_transform(PCA(n_components=nc).fit_transform(np.asarray(data)))
    
    label_colors = [color_map(i/max_label) for i in labels]

    return tsne[:,0], tsne[:,1], label_colors