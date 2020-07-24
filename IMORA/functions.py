import operator
import functools
import copy
from collections import Counter
import numpy as np

import seaborn as sns
import scipy.spatial.distance as scipy_dist
import matplotlib.pyplot as plt
from sklearn.exceptions import NotFittedError

from typing import List, Union
from IMORA.ruleset import RuleSet
# from IMORA.rule import Rule


def find_bins(x, nb_bucket):
    """
    Function used to find the bins to discretize xcol in nb_bucket modalities

    Parameters
    ----------
    x : {Series type}
           Serie to discretize

    nb_bucket : {int type}
                Number of modalities

    Return
    ------
    bins : {ndarray type}
           The bins for disretization (result from numpy percentile function)
    """
    # Find the bins for nb_bucket
    q_list = np.arange(100.0 / nb_bucket, 100.0, 100.0 / nb_bucket)
    bins = np.array([np.nanpercentile(x, i) for i in q_list])

    if bins.min() != 0:
        test_bins = bins / bins.min()
    else:
        test_bins = bins

    # Test if we have same bins...
    while len(set(test_bins.round(5))) != len(bins):
        # Try to decrease the number of bucket to have unique bins
        nb_bucket -= 1
        q_list = np.arange(100.0 / nb_bucket, 100.0, 100.0 / nb_bucket)
        bins = np.array([np.nanpercentile(x, i) for i in q_list])
        if bins.min() != 0:
            test_bins = bins / bins.min()
        else:
            test_bins = bins

    return bins


def discretize(x, nb_bucket, bins=None):
    """
    Function used to have discretize xcol in nb_bucket values
    if xcol is a real series and do nothing if xcol is a string series

    Parameters
    ----------
    x : {Series type}
           Series to discretize

    nb_bucket : {int type}
                Number of modalities

    bins : {ndarray type}, optional, default None
           If you have already calculate the bins for xcol

    Return
    ------
    x_discretized : {Series type}
                       The discretization of xcol
    """
    if np.issubdtype(x.dtype, np.floating):
        # extraction of the list of xcol values
        notnan_vector = np.extract(np.isfinite(x), x)
        nan_index = ~np.isfinite(x)
        # Test if xcol have more than nb_bucket different values
        if len(set(notnan_vector)) >= nb_bucket or bins is not None:
            if bins is None:
                bins = find_bins(x, nb_bucket)
            # discretization of the xcol with bins
            x_discretized = np.digitize(x, bins=bins)
            x_discretized = np.array(x_discretized, dtype='float')

            if sum(nan_index) > 0:
                x_discretized[nan_index] = np.nan

            return x_discretized

        return x

    else:
        return x


def select_rs(rs, gamma=1.0, selected_rs=None):
    """
    Returns a subset of a given rs. This subset is seeking by
    minimization/maximization of the criterion on the training set
    """
    # Then optimization
    if selected_rs is None or len(selected_rs) == 0:
        selected_rs = RuleSet(rs[:1])
        id_rule = 1
    else:
        id_rule = 0

    nb_rules = len(rs)

    for i in range(id_rule, nb_rules):
        rs_copy = copy.deepcopy(selected_rs)
        new_rules = rs[i]
        # Test union criteria for each rule in the current selected RuleSet
        utest = [new_rules.union_test(rule.get_activation(), gamma) for rule in rs_copy]
        if all(utest) and new_rules.union_test(selected_rs.calc_activation(), gamma):
            selected_rs.append(new_rules)

    return selected_rs


def check_is_fitted(estimator):
    if len(estimator.rules_list) == 0:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this estimator.")
        raise NotFittedError(msg % {'name': type(estimator).__name__})


def inter(rs) -> int:
    return sum(map(lambda r: r.length, rs))


def mse_function(prediction_vector: np.ndarray, y: np.ndarray):
    """
    Compute the mean squared error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} (\\hat{y}_i - y_i)^2 $"

    Parameters
    ----------
    prediction_vector : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type}
        The real target values (real numbers)

    Return
    ------
    criterion : {float type}
           the mean squared error
    """
    assert len(prediction_vector) == len(y), \
        'The two array must have the same length'
    error_vector = prediction_vector - y
    criterion = np.nanmean(error_vector ** 2)
    return criterion


def mae_function(prediction_vector: np.ndarray, y: np.ndarray):
    """
    Compute the mean absolute error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} |\\hat{y}_i - y_i| $"

    Parameters
    ----------
    prediction_vector : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type}
        The real target values (real numbers)

    Return
    ------
    criterion : {float type}
           the mean absolute error
    """
    assert len(prediction_vector) == len(y), \
        'The two array must have the same length'
    error_vect = np.abs(prediction_vector - y)
    criterion = np.nanmean(error_vect)
    return criterion


def aae_function(prediction_vector: np.ndarray, y: np.ndarray):
    """
    Compute the mean squared error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} (\\hat{y}_i - y_i)$"

    Parameters
    ----------
    prediction_vector : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type}
        The real target values (real numbers)

    Return
    ------
    criterion : {float type}
           the mean squared error
    """
    assert len(prediction_vector) == len(y), \
        'The two array must have the same length'
    error_vector = np.mean(np.abs(prediction_vector - y))
    median_error = np.mean(np.abs(y - np.median(y)))
    return error_vector / median_error


def make_condition(rule):
    """
    Evaluate all suitable rules (i.e satisfying all criteria)
    on a given feature.
    Parameters
    ----------
    rule : {rule type}
           A rule

    Return
    ------
    conditions_str : {str type}
                     A new string for the condition of the rule
    """
    conditions = rule.get_param('conditions').get_attr()
    length = rule.get_param('length')
    conditions_str = ''
    for i in range(length):
        if i > 0:
            conditions_str += ' & '

        conditions_str += conditions[0][i]
        if conditions[2][i] == conditions[3][i]:
            conditions_str += ' = '
            conditions_str += str(conditions[2][i])
        else:
            conditions_str += r' $\in$ ['
            conditions_str += str(conditions[2][i])
            conditions_str += ', '
            conditions_str += str(conditions[3][i])
            conditions_str += ']'

    return conditions_str


def calc_coverage(vect: np.ndarray):
    """
    Compute the coverage rate of an activation vector

    Parameters
    ----------
    vect : {array type}
           A activation vector. It means a sparse array with two
           different values 0, if the rule is not active
           and the 1 is the rule is active.

    Return
    ------
    cov : {float type}
          The coverage rate
    """
    u = np.sign(vect)
    return np.dot(u, u) / float(u.size)


def calc_prediction(activation_vector: np.ndarray, y: np.ndarray):
    """
    Compute the empirical conditional expectation of y
    knowing x

    Parameters
    ----------
    activation_vector : {array type}
                  A activation vector. It means a sparse array with two
                  different values 0, if the rule is not active
                  and the 1 is the rule is active.

    y : {array type}
        The target values (real numbers)

    Return
    ------
    predictions : {float type}
           The empirical conditional expectation of y
           knowing x
    """
    y_cond = np.extract(activation_vector, y)
    if sum(~np.isnan(y_cond)) == 0:
        return 0
    else:
        predictions = np.nanmean(y_cond)
        return predictions


def calc_variance(activation_vector: np.ndarray, y: np.ndarray):
    """
    Compute the empirical conditional expectation of y
    knowing x

    Parameters
    ----------
    activation_vector : {array type}
                  A activation vector. It means a sparse array with two
                  different values 0, if the rule is not active
                  and the 1 is the rule is active.

    y : {array type}
        The target values (real numbers)

    Return
    ------
    cond_var : {float type}
               The empirical conditional variance of y
               knowing x
    """
    # cov = calc_coverage(activation_vector)
    # y_cond = activation_vector * y
    # cond_var = 1. / cov * (np.mean(y_cond ** 2) - 1. / cov * np.mean(y_cond) ** 2)
    sub_y = np.extract(activation_vector, y)
    cond_var = np.var(sub_y)

    return cond_var


def calc_criterion(pred: float, y: np.ndarray, method: str = 'mse'):
    """
    Compute the criteria

    Parameters
    ----------
    pred : {float type}

    y : {array type}
        The real target values (real numbers)

    method : {string type}
             The method mse_function or mse_function criterion

    Return
    ------
    criterion : {float type}
           Criteria value
    """
    criterion = []
    for i in range(y.ndim):
        prediction_vector = pred[i] * y[:, i].astype('bool')

        if method == 'mse':
            criterion.append(mse_function(prediction_vector, y[:, i]))

        elif method == 'mae':
            criterion.append(mae_function(prediction_vector, y[:, i]))

        elif method == 'aae':
            criterion.append(aae_function(prediction_vector, y[:, i]))

        else:
            raise 'Method %s unknown' % method

    return criterion


def dist(u: np.ndarray, v: np.ndarray):
    """
    Compute the distance between two prediction vector

    Parameters
    ----------
    u,v : {array type}
          A predictor vector. It means a sparse array with two
          different values 0, if the rule is not active
          and the prediction is the rule is active.

    Return
    ------
    Distance between u and v
    """
    assert len(u) == len(v), \
        'The two array must have the same length'
    u = np.sign(u)
    v = np.sign(v)
    num = np.dot(u, v)
    deno = min(np.dot(u, u),
               np.dot(v, v))
    return 1 - num / deno


def get_variables_count(ruleset):
    """
    Get a counter of all different features in the ruleset

    Parameters
    ----------
    ruleset : {ruleset type}
             A set of rules

    Return
    ------
    count : {Counter type}
            Counter of all different features in the ruleset
    """
    col_varuleset = [rule.conditions.get_param('features_name')
                     for rule in ruleset]
    varuleset_list = functools.reduce(operator.add, col_varuleset)
    count = Counter(varuleset_list)

    count = count.most_common()
    return count


def make_selected_df(ruleset):
    df = ruleset.to_df()

    df.rename(columns={"Cov": "Coverage", "Pred": "Prediction",
                       'Var': 'Variance', 'Crit': 'Criterion'},
              inplace=True)

    df['Conditions'] = [make_condition(rule) for rule in ruleset]
    selected_df = df[['Conditions', 'Coverage',
                      'Prediction', 'Variance',
                      'Criterion']].copy()

    selected_df['Coverage'] = selected_df.Coverage.round(2)
    selected_df['Prediction'] = selected_df.Prediction.round(2)
    selected_df['Variance'] = selected_df.Variance.round(2)
    selected_df['Criterion'] = selected_df.Criterion.round(2)

    return selected_df


def plot_counter_variables(ruleset, nb_max: int = None):
    counter = get_variables_count(ruleset)

    x_labels = list(map(lambda item: item[0], counter))
    values = list(map(lambda item: item[1], counter))

    fig = plt.figure()
    ax = plt.subplot()

    if nb_max is not None:
        x_labels = x_labels[:nb_max]
        values = values[:nb_max]

    g = sns.barplot(y=x_labels, x=values, ax=ax, ci=None)
    g.set(xlim=(0, max(values) + 1), ylabel='Variable', xlabel='Count')

    return fig


def plot_dist(ruleset, x: np.ndarray = None):
    rules_names = ruleset.get_rules_name()

    predictions_vector_list = [rule.get_predictions_vector(x) for rule in ruleset]
    predictions_matrix = np.array(predictions_vector_list)

    distance_vector = scipy_dist.pdist(predictions_matrix, metric=dist)
    distance_matrix = scipy_dist.squareform(distance_vector)

    # Set up the matplotlib figure
    f = plt.figure()
    ax = plt.subplot()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(distance_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    vmax = np.max(distance_matrix)
    vmin = np.min(distance_matrix)
    # center = np.mean(distance_matrix)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(distance_matrix, cmap=cmap, ax=ax,
                vmax=vmax, vmin=vmin, center=1.,
                square=True, xticklabels=rules_names,
                yticklabels=rules_names, mask=mask)

    plt.yticks(rotation=0)
    plt.xticks(rotation=90)

    return f
