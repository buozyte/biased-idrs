import time

import numpy as np
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr


def input_dependent_certified_radius_given_pb(sigma_0, r, dim, p_b, num_steps, r_computer=None, safe_option=False):
    """
    Computation of the certified radius.

    :param sigma_0: base sigma
    :param r: semi-elasticity constant for chosen sigma function
    :param dim: dimension of the input data space
    :param p_b: certain probability (see paper/expose)
    :param num_steps: number of steps to take in the computation in the radius
    :param r_computer: ?
    :param safe_option: indicator wether to compute the largest radius for both (</>) cases (or only for one)
    :return: certified radius for the current point
    """

    numpy2ri.activate()
    if r_computer is None:
        r_computer = importr('stats')

    dists = np.linspace(start=10 * sigma_0 / num_steps, stop=10 * sigma_0, num=num_steps)
    i = 0

    # compute "true sigma"
    # --- CIFAR10 THRESHOLD --------------------------
    # sigma_t = 0.9993 + 0.001 * np.log10(p_b)
    # ------------------------------------------------
    # ---MNIST THRESHOLD -----------------------------
    sigma_t = 0.9988 + 0.001 * np.log10(p_b)
    # ------------------------------------------------
    sigma_t = np.zeros_like(dists) + sigma_t * sigma_0  # TODO: why *sigma_0?
    sigmas_1_1 = np.exp(-r * dists) * sigma_0  # compute exact sigma_1
    true_sigmas = np.minimum(sigma_t, sigmas_1_1)

    loc_radii = np.zeros(len(dists))
    loc_xi_bigger = np.zeros(len(dists))
    start = time.time()

    # r_computer.qchisq(probabilities [array], degrees of freedom, non-centrality parameter)
    #    -> compute value of chi squared quantile function
    loc_radii[i] = np.asarray(
        r_computer.qchisq(p_b, dim, sigma_0 ** 2 / (sigma_0 ** 2 - true_sigmas[i] ** 2) ** 2 * dists[i] ** 2))

    # r_computer.pchisq(quantiles [array], degrees of freedom, non-centrality parameter
    #    -> compute value of chi squared cdf function
    non_centrality = true_sigmas[i] ** 2 / (sigma_0 ** 2 - true_sigmas[i] ** 2) ** 2 * dists[i] ** 2
    loc_xi_bigger[i] = np.asarray(
        r_computer.pchisq(loc_radii[i] * sigma_0 ** 2 / true_sigmas[i] ** 2, dim, non_centrality))

    # if xi function at distance 0 already above 0.5, then no distance will be certified
    if loc_xi_bigger[i] >= 0.5:
        certified_radius = 0
    else:
        j = i
        # find base so that the value at index b**2 will lead to a probability larger than 0.5
        # -> the radius will be between the index (b-1)**2 and b**2 (-> faster limitation of potential radii)
        while loc_xi_bigger[j] < 0.5 and (i + 1) ** 2 < len(dists):
            i += 1
            j = i ** 2
            loc_radii[j] = np.asarray(
                r_computer.qchisq(p_b, dim, sigma_0 ** 2 / (sigma_0 ** 2 - true_sigmas[j] ** 2) ** 2 * dists[j] ** 2))
            non_centrality = true_sigmas[i] ** 2 / (sigma_0 ** 2 - true_sigmas[i] ** 2) ** 2 * dists[i] ** 2
            loc_xi_bigger[j] = np.asarray(
                r_computer.pchisq(loc_radii[j] * sigma_0 ** 2 / true_sigmas[j] ** 2, dim, non_centrality))
        # reset the index to the last "valid" index
        # -> traverse through values in "correct" order, starting at the value at index (b-1)**2
        i = (i - 1) ** 2
        while loc_xi_bigger[i] < 0.5 and i + 1 < len(dists):
            i += 1
            loc_radii[i] = np.asarray(
                r_computer.qchisq(p_b, dim, sigma_0 ** 2 / (sigma_0 ** 2 - true_sigmas[i] ** 2) ** 2 * dists[i] ** 2))
            non_centrality = true_sigmas[i] ** 2 / (sigma_0 ** 2 - true_sigmas[i] ** 2) ** 2 * dists[i] ** 2
            loc_xi_bigger[i] = np.asarray(
                r_computer.pchisq(loc_radii[i] * sigma_0 ** 2 / true_sigmas[i] ** 2, dim, non_centrality))
        certified_radius = dists[i - 1]

    # do the same as above, but with the "upper" threshold -> actual certified radius is the smaller one (of the two)
    if safe_option:
        i_2 = 0
        p_a = 1 - p_b

        # compute "true sigma"
        sigma_T = 1 / sigma_t
        sigma_T = np.zeros_like(dists) + sigma_T * sigma_0
        sigmas_1_2 = np.exp(r * dists) * sigma_0
        true_sigmas = np.maximum(sigma_T, sigmas_1_2)

        loc_radii_2 = np.zeros(len(dists))
        loc_xi_lower = np.zeros(len(dists))
        loc_radii_2[i_2] = np.asarray(
            r_computer.qchisq(p_a, dim, sigma_0 ** 2 / (sigma_0 ** 2 - true_sigmas[i_2] ** 2) ** 2 * dists[i_2] ** 2))
        non_centrality = true_sigmas[i] ** 2 / (sigma_0 ** 2 - true_sigmas[i] ** 2) ** 2 * dists[i] ** 2
        loc_xi_lower[i_2] = 1 - np.asarray(
            r_computer.pchisq(loc_radii_2[i_2] * sigma_0 ** 2 / true_sigmas[i_2] ** 2, dim, non_centrality))
        if loc_xi_lower[i_2] >= 0.5:
            certified_radius_2 = 0
        else:
            j_2 = i_2
            while loc_xi_lower[j_2] < 0.5 and (i_2 + 1) ** 2 < len(dists):
                i_2 += 1
                j_2 = i_2 ** 2
                loc_radii_2[j_2] = np.asarray(
                    r_computer.qchisq(p_a, dim,
                                      sigma_0 ** 2 / (sigma_0 ** 2 - sigmas_1_2[j_2] ** 2) ** 2 * dists[j_2] ** 2))
                non_centrality = true_sigmas[i] ** 2 / (sigma_0 ** 2 - true_sigmas[i] ** 2) ** 2 * dists[i] ** 2
                loc_xi_lower[j_2] = 1 - np.asarray(
                    r_computer.pchisq(loc_radii_2[j_2] * sigma_0 ** 2 / true_sigmas[j_2] ** 2, dim, non_centrality))
            i_2 = (i_2 - 1) ** 2
            while loc_xi_lower[i_2] < 0.5 and i_2 + 1 < len(dists):
                i_2 += 1
                loc_radii_2[i_2] = np.asarray(
                    r_computer.qchisq(p_a, dim,
                                      sigma_0 ** 2 / (sigma_0 ** 2 - sigmas_1_2[i_2] ** 2) ** 2 * dists[i_2] ** 2))
                non_centrality = true_sigmas[i] ** 2 / (sigma_0 ** 2 - true_sigmas[i] ** 2) ** 2 * dists[i] ** 2
                loc_xi_lower[i_2] = 1 - np.asarray(
                    r_computer.pchisq(loc_radii_2[i_2] * sigma_0 ** 2 / true_sigmas[i_2] ** 2, dim, non_centrality))
            certified_radius_2 = dists[i_2 - 1]
        certified_radius = min(certified_radius, certified_radius_2)

    numpy2ri.deactivate()
    end = time.time()
    return certified_radius
