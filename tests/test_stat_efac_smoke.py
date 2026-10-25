"""End-to-end smoke tests for simulate(stat_efac=True) (EFAC0 feature, branch stat-efac).

Runs sterne's full Bayesian machinery on a tiny 6-epoch synthetic pmpar.in
(zero proper motion / parallax truth; efac inference only needs the error
model) with small iteration/walker counts so the whole file stays well under
two minutes.  Requires the TEMPO2 runtime ($TEMPO2 + T2runtime ephemeris
DE421) -- skipped automatically otherwise.

Scenario directions: EFAC0 estimates the ratio (true scatter)/(reported error
columns), because errs_new = efac * errs.  The mis-scaled scenario therefore
draws positions with noise 1.5x the reported columns (an under-reported error
budget), which targets efac_0 ~ 1.5.  (Inflating the error columns themselves
by 1.5 with unchanged noise would instead drive efac_0 to ~1/1.5 by the
change-of-variables symmetry of the likelihood.)

All inputs and outputs (inits, pmpar.in, bilby outdir) live in tmp_path;
nothing is written into the repository tree.
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")  # headless corner plots

import numpy as np
import pytest
from astropy.table import Table

_TEMPO2 = os.getenv("TEMPO2")


def _tempo2_runtime_ok():
    """$TEMPO2 set, DE421 ephemeris present, and the novas ephemeris machinery
    functional (some novas builds ship a libnovas.so without the eph_manager
    symbols, which would only fail deep inside positions())."""
    if not _TEMPO2:
        return False
    eph = os.path.join(_TEMPO2, "T2runtime", "ephemeris", "DE421.1950.2050")
    if not os.path.exists(eph):
        return False
    try:
        from novas.compat.eph_manager import ephem_open

        ephem_open(eph)
        return True
    except Exception:
        return False


TEMPO2_OK = _tempo2_runtime_ok()

pytestmark = pytest.mark.skipif(
    not TEMPO2_OK,
    reason="TEMPO2 runtime ($TEMPO2/T2runtime, DE421 + novas) not available",
)

# --- synthetic series definition -------------------------------------------------
REFEPOCH = 60646.72  # MJD, mid-series
RA_H, RA_M, RA_S = 18, 39, 1.060
DEC_D, DEC_M, DEC_S = -1, 0, 5.0  # dec-seconds kept away from 0 so noise cannot flip its sign
EPOCHS = [60464.0, 60537.0, 60610.0, 60683.0, 60756.0, 60829.0]  # 6 epochs over ~1 yr
RAERR_S = 1.0e-4  # seconds of time (~1.5 mas on sky)
DECERR_AS = 1.0e-3  # arcsec (1 mas)
SEED = 20261026

ITERATIONS = 500
NWALKERS = 30

# shares for one pmparin, 9 rows in the alphabetical root order
# (dec, efac, efad, incl, mu_a, mu_d, om_asc, px, ra); incl/om_asc need a
# parfile, efad needs preliminaries -> all off; px on keeps the run realistic.
EFAC0_SHARES = [[0], [0], [-1], [-1], [0], [0], [-1], [0], [0]]


def _ra_rad():
    return (RA_H + RA_M / 60.0 + RA_S / 3600.0) * 15.0 * np.pi / 180.0


def _dec_rad():
    return -abs(DEC_D) * np.pi / 180.0 - (DEC_M / 60.0 + DEC_S / 3600.0) * np.pi / 180.0


def _noise_deviates():
    rng = np.random.default_rng(SEED)
    ra_dev = rng.normal(0.0, 1.0, len(EPOCHS))
    dec_dev = rng.normal(0.0, 1.0, len(EPOCHS))
    return ra_dev, dec_dev


def _write_pmparin(path, noise_scale=1.0):
    """Write the synthetic pmpar.in: constant truth + Gaussian noise.

    `noise_scale` multiplies the noise while the error columns stay at the
    fiducial scale (noise_scale=1.5 == error budget under-reported by 1.5x).
    """
    ra_dev, dec_dev = _noise_deviates()
    with open(path, "w") as f:
        f.write("# synthetic 6-epoch series for the stat_efac smoke test\n")
        f.write("# truth: constant position (zero pm, zero px)\n")
        f.write("epoch = %f\n" % REFEPOCH)
        for i, mjd in enumerate(EPOCHS):
            ra_s = RA_S + noise_scale * RAERR_S * ra_dev[i]
            dec_s = DEC_S + noise_scale * DECERR_AS * dec_dev[i]
            f.write(
                "%f %02d:%02d:%012.9f %.7f %02d:%02d:%012.9f %.7f\n"
                % (
                    mjd, RA_H, RA_M, ra_s, RAERR_S,
                    DEC_D, DEC_M, dec_s, DECERR_AS,
                )
            )
    return str(path)


def _write_inits(path, with_efac):
    """Hand-made .inits (generate_initsfile needs the pmpar binary)."""
    lines = [
        "# priors for the stat_efac smoke test",
        "# units: dec/ra rad; px mas; mu_a/mu_d mas/yr",
        "dec_0: %.12f,%.12f,Uniform" % (_dec_rad() - 2.5e-7, _dec_rad() + 2.5e-7),
        "ra_0: %.12f,%.12f,Uniform" % (_ra_rad() - 2.5e-7, _ra_rad() + 2.5e-7),
        "mu_a_0: -100,100,Uniform",
        "mu_d_0: -100,100,Uniform",
        "px_0: -100,100,Uniform",
    ]
    if with_efac:
        lines.append("efac_0: 0,15,Uniform")  # sterne default efac prior
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return str(path)


def _run_simulate(tmp_path, tag, noise_scale=1.0, stat_efac=False):
    from sterne.simulate import simulate  # deferred: heavy import at skip/collection time

    outdir = tmp_path / tag
    outdir.mkdir()
    pmparin = _write_pmparin(outdir / "synthetic.pmpar.in", noise_scale)
    inits = _write_inits(outdir / "synthetic.inits", with_efac=stat_efac)
    np.random.seed(SEED)  # deterministic walker initialization / emcee stream
    simulate(
        REFEPOCH, inits, pmparin, "",
        shares=[list(row) for row in EFAC0_SHARES],
        outdir=str(outdir), iterations=ITERATIONS, nwalkers=NWALKERS,
        stat_efac=stat_efac,
    )
    return Table.read(str(outdir / "posterior_samples.dat"), format="ascii")


def test_stat_efac_correct_errors(tmp_path):
    """Correctly scaled error columns -> posterior efac_0 consistent with 1."""
    samples = _run_simulate(tmp_path, "efac0_correct", stat_efac=True)
    assert "efac_0" in samples.colnames
    median = float(np.median(samples["efac_0"]))
    assert 0.8 <= median <= 1.3, "efac_0 median %.3f outside [0.8, 1.3]" % median


def test_stat_efac_underreported_errors_x1p5(tmp_path):
    """Noise 1.5x the error columns (mis-scaled budget) -> efac_0 ~ 1.5."""
    samples = _run_simulate(tmp_path, "efac0_x15", noise_scale=1.5, stat_efac=True)
    assert "efac_0" in samples.colnames
    median = float(np.median(samples["efac_0"]))
    assert 1.2 <= median <= 2.0, "efac_0 median %.3f outside [1.2, 2.0]" % median


def test_efac_off_control(tmp_path):
    """Without stat_efac, the efac share row is force-disabled (pre-feature behavior)."""
    samples = _run_simulate(tmp_path, "efac_off_control", stat_efac=False)
    assert not any("efac" in name for name in samples.colnames)
    assert "px_0" in samples.colnames  # the astrometric fit itself ran
