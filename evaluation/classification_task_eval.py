import os
import torch
from PIL import Image
from scipy import stats
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import metrics as sklmetrics
import pytorch_lightning.metrics.functional.classification as metrics



def save_conf_mat(predictions, labels, current_epoch=None, current_run=None, mode=None, bucket_mode=None):
    """
        Saves confusion matrix  as .png
    """
    if bucket_mode == 0.1 or bucket_mode == 0.2:
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        predictions = le.transform(predictions)
        labels = le.transform(labels)

    matrix = sklmetrics.confusion_matrix(torch.Tensor(labels), torch.Tensor(predictions))
    plt.imshow(matrix, cmap='Blues', interpolation='none')
    if mode=='eval':
        plt.title(f'Epoch {current_epoch}', fontsize=30)
    if mode=='test':
        plt.title(f'Test_Confusion Matrix', fontsize=30)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if mode=='test':
        plt.savefig('test_confusion_matrix.png')
        im = Image.open('./test_confusion_matrix.png')
        im.save("test.gif", save_all=True, append_images=[im], duration=1000, loop=0)

    if mode=='eval':
        if current_epoch==0:
            plt.savefig(f'./image_{current_epoch}.png') # needs to be named like this for the gif
        else:
            plt.savefig(f'./image_{current_epoch}_{current_run}.png')


def prepare_gif():
    """
        Creates dictionary of confusion matrices over val epochs to be used for a GIF creation
    """
    image_dict = {}
    for filename in os.listdir('.'):
        if filename.endswith('.png'):
            image_dict[filename] = Image.open(f'{filename}')
    image_dict = dict(sorted(image_dict.items()))
    images = list(image_dict.values())
    image_dict['image_0.png'].save("out.gif", save_all=True, append_images=images, duration=1000, loop=0)


def classification_metrics(predictions, labels):
    """
    Computes classification metrics.
    Args:
        predictions of an epoch and corresponding labels
    Returns:
        dictionary comprising of classification metrics
    """
    predictions = torch.Tensor(predictions)
    labels = torch.Tensor(labels)

    pcc = stats.pearsonr(predictions, labels)
    accuracy = metrics.accuracy(predictions, labels)
    macro_accuracy = metrics.accuracy(
        predictions, labels, class_reduction='macro')
    weighted_accuracy = metrics.accuracy(
        predictions, labels, class_reduction='weighted')
    precision = metrics.precision(
        predictions, labels, class_reduction='micro')
    macro_precision = metrics.precision(
        predictions, labels, class_reduction='macro')
    weighted_precision = metrics.precision(
        predictions, labels, class_reduction='weighted')

    recall = metrics.recall(
        predictions, labels, class_reduction='micro')
    macro_recall = metrics.recall(
        predictions, labels, class_reduction='macro')
    weighted_recall = metrics.recall(
        predictions, labels, class_reduction='weighted')

    return {'accuracy': accuracy, 'macro_accuracy': macro_accuracy, 'weighted_accuracy': weighted_accuracy,
            'precision': precision, 'macro_precision': macro_precision, 'weighted_precision': weighted_precision,
            'recall': recall, 'macro_recall': macro_recall, 'weighted_recall': weighted_recall,
            'pearson_cor_coef': torch.Tensor([pcc[0]])
            }
