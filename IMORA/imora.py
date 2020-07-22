
import copy
from typing import List

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor, _tree
from sklearn.utils.validation import check_X_y

from IMORA.rule import Rule
from IMORA.ruleconditions import RuleConditions


def _fit_estimator(estimator, X, y, **fit_params):
    estimator = clone(estimator)
    estimator.fit(X, y, **fit_params)
    return estimator


def _eval_rules(rule: Rule, y: np.ndarray, X: np.ndarray):
    """
    Parameters
    ----------
    rule: rule to evaluate
    y: variable of interest
    X: features matrix

    Returns
    -------
    rule: rule evaluated on (X, y)
    """
    rule.calc_stats(x=X, y=y)
    return rule


def _extract_rules_from_tree(tree: DecisionTreeRegressor,
                            features: List[str],
                            xmin: List[float],
                            xmax: List[float],
                            get_leaf: bool = False) -> List[Rule]:
    dt = tree.tree_

    def visitor(node, depth, cond=None, rule_list=None):
        if rule_list is None:
            rule_list = []
        if dt.feature[node] != _tree.TREE_UNDEFINED:
            # If
            new_cond = RuleConditions([features[dt.feature[node]]],
                                      [dt.feature[node]],
                                      bmin=[xmin[dt.feature[node]]],
                                      bmax=[dt.threshold[node]],
                                      xmin=[xmin[dt.feature[node]]],
                                      xmax=[xmax[dt.feature[node]]])
            if cond is not None:
                if dt.feature[node] not in cond.features_index:
                    conditions_list = list(map(lambda c1, c2: c1 + c2, cond.get_attr(),
                                               new_cond.get_attr()))

                    new_cond = RuleConditions(features_name=conditions_list[0],
                                              features_index=conditions_list[1],
                                              bmin=conditions_list[2],
                                              bmax=conditions_list[3],
                                              xmax=conditions_list[5],
                                              xmin=conditions_list[4])
                else:
                    new_bmax = dt.threshold[node]
                    new_cond = copy.deepcopy(cond)
                    place = cond.features_index.index(dt.feature[node])
                    new_cond.bmax[place] = min(new_bmax, new_cond.bmax[place])

            # print (Rule(new_cond))
            new_rg = Rule(copy.deepcopy(new_cond))
            if get_leaf is False:
                rule_list.append(new_rg)

            rule_list = visitor(dt.children_left[node], depth + 1,
                                new_cond, rule_list)

            # Else
            new_cond = RuleConditions([features[dt.feature[node]]],
                                      [dt.feature[node]],
                                      bmin=[dt.threshold[node]],
                                      bmax=[xmax[dt.feature[node]]],
                                      xmin=[xmin[dt.feature[node]]],
                                      xmax=[xmax[dt.feature[node]]])
            if cond is not None:
                if dt.feature[node] not in cond.features_index:
                    conditions_list = list(map(lambda c1, c2: c1 + c2, cond.get_attr(),
                                               new_cond.get_attr()))
                    new_cond = RuleConditions(features_name=conditions_list[0],
                                              features_index=conditions_list[1],
                                              bmin=conditions_list[2],
                                              bmax=conditions_list[3],
                                              xmax=conditions_list[5],
                                              xmin=conditions_list[4])
                else:
                    new_bmin = dt.threshold[node]
                    new_bmax = xmax[dt.feature[node]]
                    new_cond = copy.deepcopy(cond)
                    place = new_cond.features_index.index(dt.feature[node])
                    new_cond.bmin[place] = max(new_bmin, new_cond.bmin[place])
                    new_cond.bmax[place] = max(new_bmax, new_cond.bmax[place])

            new_rg = Rule(copy.deepcopy(new_cond))
            if get_leaf is False:
                rule_list.append(new_rg)

            rule_list = visitor(dt.children_right[node], depth + 1, new_cond, rule_list)

        elif get_leaf:
            rule_list.append(Rule(copy.deepcopy(cond)))

        return rule_list

    rule_list = visitor(0, 1)
    return rule_list


class IMORA:
    """
    ...
    """

    def __init__(self, n_jobs: int = 1):
        self.n_jobs = n_jobs
        self.estimator = DecisionTreeRegressor()
        self.rule_dict = {}
        self.dict_keys = []
        self.estimators_ = []

    def fit(self, X, y, features: List[str] = None):
        X, y = check_X_y(X, y, ensure_min_samples=10, accept_sparse=True,
                         multi_output=True, force_all_finite='allow-nan', y_numeric=True)

        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions for "
                             "multi-output regression but has only one.")
        else:
            self.dict_keys = ['Cp_' + str(i) for i in range(y.ndim)] # Cp for component
        x_min = X.min(axis=0)
        x_max = X.max(axis=0)
        if features is None:
            self.features = ['feature_' + str(col) for col in range(0, X.shape[1])]
        else:
            self.features = features

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(self.estimator, X, y[:, i])
            for i in range(y.shape[1]))
        self.extract_rules(x_min, x_max)
        self.eval_rules(X, y)
        self.select_rules()

    def extract_rules(self, x_min: List[float], x_max: List[float]):
        dim = 0
        for tree in self.estimators_:
            self.rule_dict[self.dict_keys[dim]] = _extract_rules_from_tree(tree, self.features,
                                                                           x_min, x_max)
            dim += 1

    def eval_rules(self, X: np.ndarray, y: np.ndarray):
        for dim in range(y.ndim):
            self.rule_dict[self.dict_keys[dim]] = Parallel(n_jobs=self.n_jobs,
                                                        backend="multiprocessing")(
                delayed(_eval_rules)(rule, y[:, dim], X)
                for rule in self.rule_dict[self.dict_keys[dim]])

    def select_rules(self):
        pass

    def predict(self, X):
        pass