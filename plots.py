import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn.manifold import TSNE


def plot_confusion(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='coolwarm'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_tsne(df, labels, title=""):
    """
    Plot the t-SNE embedding
    
    Positional arguments:
        df - t-SNE fitted data
        labels - labels corresponding to df
    """
    unique_labels = np.unique(labels)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'SaddleBrown', 'SlateGrey', 'DarkOrange']
    
    for i in range(len(unique_labels)):
        # Mask and filter the array
        m = labels==unique_labels[i]
        m_df = df[m]

        # Draw the points
        plt.scatter(m_df[:,0], m_df[:,1], label=unique_labels[i], alpha=0.5, color=colors[i], marker='.')
        
    plt.legend(ncol=2, scatterpoints=1)
    plt.xlabel("First Component")
    plt.ylabel("Second Component")
    plt.title("Data Projection into 2D Subspace - t-SNE{}".format(title))
