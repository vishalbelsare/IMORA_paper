"""
...
"""
import copy
from typing import List

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor, _tree
from sklearn.utils.validation import check_X_y, check_array

from IMORA.functions import select_rs, find_bins, discretize, check_is_fitted
from IMORA.ruleset import RuleSet
from IMORA.rule import Rule
from IMORA.ruleconditions import RuleConditions


def _fit_estimator(estimator, X: np.ndarray, y: np.ndarray, **fit_params):
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


def _occurrences_count(rule: Rule, rules_list: List[Rule]):
    noc = rules_list.count(rule)
    rule.set_params(noc=noc)
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


def _calc_pred(rule, X, weigths):
    act = rule.calc_activation(X)
    pred = [act * weigths * p for p in rule.pred]
    pred = np.array(pred).T
    pred[pred == 0] = np.nan
    return pred


def _discretize(xcol: np.ndarray, var_name: str, nb_bins: int, bins_dict: dict):
    try:
        xcol = np.array(xcol.flat, dtype=np.float)
    except ValueError:
        xcol = np.array(xcol.flat, dtype=np.str)

    if np.issubdtype(xcol.dtype, np.floating):
        if var_name not in bins_dict:
            if len(set(xcol)) >= nb_bins:
                bins = find_bins(xcol, nb_bins)
                discretized_column = discretize(xcol, nb_bins, bins)
                bins_dict[var_name] = bins
            else:
                discretized_column = xcol
        else:
            bins = bins_dict[var_name]
            discretized_column = discretize(xcol, nb_bins, bins)
    else:
        discretized_column = xcol

    return discretized_column, bins_dict


class IMORA:
    """
    ...
    """

    def __init__(self, alpha: float = 1/2. - 0.001, gamma: float = 0.90, l_max: int = 3,
                 noc_min: int = 2, nb_bins: int = 10, n_jobs: int = -1):
        self.alpha = alpha
        self.gamma = gamma
        self.l_max = l_max
        self.noc_min = noc_min
        self.nb_bins = nb_bins
        self.n_jobs = n_jobs
        self.n = 0
        self.estimator = DecisionTreeRegressor()
        self.features = []
        self.rules_dict = {}
        self.bins_dict = {}
        self.rules_list = []
        self.select_rs = RuleSet([])
        self.dict_keys = []
        self.estimators_ = []

    def _validate_X_predict(self, X: np.ndarray, check_input: bool = True):
        """Validate X whenever one tries to predict"""
        if check_input:
            X = check_array(X, accept_sparse=True, force_all_finite='allow-nan')

        n_features = X.shape[1]
        if len(self.features) != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (len(self.features), n_features))

        return X

    def fit(self, X, y, features: List[str] = None):
        X, y = check_X_y(X, y, ensure_min_samples=10, accept_sparse=True,
                         multi_output=True, force_all_finite='allow-nan', y_numeric=True)

        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions for "
                             "multi-output regression but has only one.")
        else:
            self.dict_keys = ['Cp_' + str(i) for i in range(y.ndim)]  # Cp for component
        self.n = y.shape[0]

        if features is None:
            self.features = ['feature_' + str(col) for col in range(0, X.shape[1])]
        else:
            self.features = features

        X = self.discretize(X)
        x_min = X.min(axis=0)
        x_max = X.max(axis=0)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(self.estimator, X, y[:, i])
            for i in range(y.ndim))
        self.extract_rules(x_min, x_max)
        self.eval_rules(X, y)
        self.select_rules()

    def discretize(self, x: np.ndarray):
        """
        Used to have discrete values for each series
        to avoid float

        Parameters
        ----------
        x : {array, matrix type}, shape=[n_samples, n_features]
            Features matrix

        Return
        -------
        col : {array, matrix type}, shape=[n_samples, n_features]
              Features matrix with each features values discretized
              in nb_bucket values
        """
        discret_params = Parallel(n_jobs=self.n_jobs)(
            delayed(_discretize)(x[:, i], self.features[i], self.nb_bins, self.bins_dict)
            for i in range(x.shape[1]))

        x_mat = []
        for p in discret_params:
            x_mat.append(p[0])
            self.bins_dict.update(p[1])

        return np.array(x_mat).T

    def extract_rules(self, x_min: List[float], x_max: List[float]):
        dim = 0
        for tree in self.estimators_:
            self.rules_dict[self.dict_keys[dim]] = _extract_rules_from_tree(tree, self.features,
                                                                            x_min, x_max)
            dim += 1

    def eval_rules(self, X: np.ndarray, y: np.ndarray):
        rules_list = []
        for dim in range(y.ndim):
            rules_list += Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
                delayed(_eval_rules)(rule, y, X)
                for rule in self.rules_dict[self.dict_keys[dim]])
        rules_list = self.rule_occurencies_count(rules_list)
        self.rules_list = list(set(rules_list))

    def rule_occurencies_count(self, rules_list):
        cov_min = self.n ** (-self.alpha)
        sub_rulelist = list(filter(lambda rule: rule.length <= self.l_max, rules_list))
        sub_rulelist = list(filter(lambda rule: rule.cov >= cov_min, sub_rulelist))
        rules_list = Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
            delayed(_occurrences_count)(rule, sub_rulelist)
            for rule in sub_rulelist)
        rules_list = list(set(rules_list))
        return rules_list

    def select_rules(self):
        sub_rulelist = list(filter(lambda rule: rule.noc >= self.noc_min, self.rules_list))
        sub_rulelist.sort(key=lambda rule: (rule.noc, -sum(rule.crit)), reverse=True)

        self.select_rs = select_rs(RuleSet(sub_rulelist), self.gamma)

    def predict(self, X: np.ndarray):
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)
        X = self.discretize(X)
        # weights = self.get_weights()

        aggregated_pred = Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
            delayed(_calc_pred)(rule, X, 1.0)
            for rule in self.select_rs)
        prediction = np.nanmean(aggregated_pred, axis=0)
        return np.nan_to_num(prediction)

    def get_weights(self):
        return 1 / len(self.select_rs)
