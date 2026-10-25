"""Unit tests for sterne.simulate.adjust_errs_with_efac (EFAC0 feature, branch stat-efac).

Pure-function tests: no TEMPO2 runtime, no sampling, no I/O.
Covers the three branches of the function:
  * -999 sentinel passthrough (efac inference disabled -> errs unchanged),
  * no-preliminaries stat-scaling branch (errs_new = efac * errs; log_efac variant),
  * preliminaries two-component branch (regression: sqrt(errs_random^2 + (efac*errs_sys)^2)).
efad behavior is out of scope for this feature and is only exercised via the
-999 sentinel (autofilled), exactly as the no-prelims simulate() path sets it.
"""
import numpy as np

from sterne.simulate import adjust_errs_with_efac

# arbitrary [errRA (rad), errDEC (rad)] block, RA first as built by
# create_list_of_dict_VLBI
ERRS = np.array([1.0e-8, 2.0e-8, 1.5e-9, 4.0e-9])


def _single_series_params(efac=None):
    """Parameter dict keyed by shares-derived names for ONE series (index 0).

    Mirrors what simulate() builds for shares row-1 enabled (efac_0) plus the
    always-fitted astrometric parameters; leaving 'efac' absent reproduces the
    efac-disabled case (auto_fill_disabled_parameters then supplies -999).
    """
    params = {
        "dec_0": -1.7e-2,
        "mu_a_0": 1.0,
        "mu_d_0": -2.0,
        "px_0": 0.5,
        "ra_0": 4.88,
    }
    if efac is not None:
        params["efac_0"] = efac
    return params


def test_minus999_sentinel_passthrough_no_prelims():
    # no efac key -> filter autofills the -999 sentinel -> errs unchanged
    out = adjust_errs_with_efac({"errs": ERRS.copy()}, _single_series_params(), 0)
    np.testing.assert_allclose(out, ERRS, rtol=1e-12)


def test_stat_scaling_branch_efac2():
    out = adjust_errs_with_efac(
        {"errs": ERRS.copy()}, _single_series_params(efac=2.0), 0
    )
    np.testing.assert_allclose(out, 2.0 * ERRS, rtol=1e-12)


def test_stat_scaling_branch_log_efac():
    out = adjust_errs_with_efac(
        {"errs": ERRS.copy()}, _single_series_params(efac=np.log(2.0)), 0,
        log_efac=True,
    )
    np.testing.assert_allclose(out, 2.0 * ERRS, rtol=1e-12)


def test_prelims_two_component_branch_regression():
    errs_random = np.array([3.0e-9, 4.0e-9, 2.0e-9, 5.0e-9])
    errs_sys = np.array([6.0e-9, 5.0e-9, 7.0e-9, 1.0e-9])
    vlbi_dict = {
        "errs": ERRS.copy(),
        "errs_random": errs_random,
        "errs_sys": errs_sys,
    }
    for efac in (1.0, 2.0):  # efad stays at the -999 sentinel (not given)
        out = adjust_errs_with_efac(
            vlbi_dict, _single_series_params(efac=efac), 0
        )
        expected = np.sqrt(errs_random ** 2 + (efac * errs_sys) ** 2)
        np.testing.assert_allclose(out, expected, rtol=1e-12)
