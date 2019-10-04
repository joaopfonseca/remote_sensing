import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


def _make_confusion_matrix(y_true, y_pred, label_mapper):
    """Generates a confusion matrix. Returns 1) Pandas Dataframe with confusion
    matrix, Producer's accuracy and User's accuracy and 2) Pandas Dataframe with
    Overall accuracy, F-Score, and G-mean score."""
    labels = list(label_mapper.values())
    labels.sort()
    cm = confusion_matrix(y_true.map(label_mapper), pd.Series(y_pred).map(label_mapper), labels=labels).T
    total_h = np.sum(cm, 0)
    total_v = np.sum(cm, 1)
    total   = sum(total_h)
    tp      = cm.diagonal()
    fp      = total_v-tp
    fn      = total_h-tp
    tn      = total-(fn+tp+fp)
    spec    = tn/(fp+tn)
    ua      = tp/(total_v+1e-100)
    pa      = tp/(total_h+1e-100)
    oa      = sum(tp)/total
    fscore  = 2*((np.mean(ua)*np.mean(pa))/(np.mean(ua)+np.mean(pa)))
    gmean   = np.sqrt(np.mean(pa)*np.mean(spec))
    core_cm = pd.DataFrame(index=labels, columns=labels, data=cm)\
                .append([
                    pd.Series(data=total_h, index=labels, name='Total'),
                    pd.Series(data=pa, index=labels, name='PA')
                    ])
    core_cm['UA']    = np.append(ua, [np.nan,np.nan])
    core_cm['UA']    = core_cm['UA'].replace('nan', np.nan) # replace with .2f
    core_cm['Total'] = np.append(total_v, [total, np.nan])
    scores = pd.DataFrame(data= [oa, fscore, gmean], index=['ACCURACY', 'F-SCORE MACRO', 'G-MEAN MACRO'], columns=['Score'])
    return core_cm, scores


def reports(y_true, y_pred, target_names):
    clf_report = classification_report(y_true.astype(int), y_pred.astype(int), target_names=target_names)
    conf_matrix, scores = _make_confusion_matrix(y_true, y_pred, target_names)
    return clf_report, conf_matrix, scores
