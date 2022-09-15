from metrics import *


def eval_pos_neg(model, test_data, metric_names, k=10, batch_size=None):
    """
    计算top-k算法（召回阶段）的performance。
    测试数据中必须包含>=k个的负样本和一个正样本。
    """
    pred_y = - model.predict(test_data, batch_size)
    return eval_rank(pred_y, metric_names, k)


def eval_rank(pred_y, metric_names, k=10):
    rank = pred_y.argsort().argsort()[:, 0]
    res_dict = {}
    for name in metric_names:
        if name == 'hr':
            res = hr(rank, k)
        elif name == 'ndcg':
            res = ndcg(rank, k)
        elif name == 'mrr':
            res = mrr(rank, k)
        else:
            break
        res_dict[name] = res
    return res_dict
