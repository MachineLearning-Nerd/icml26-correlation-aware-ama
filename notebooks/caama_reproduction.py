import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Correlation-aware auction payments: reproduced evidence

    **Headline result.** On the paper's full 3-bidder × 10-item
    Dirichlet setting, five CPU training seeds produced CA-AMA revenue
    **3.7359** versus **3.0530** for the baseline. The paper reports
    3.6205 versus 3.1363.

    This notebook embeds the completed evidence. It does not rerun the
    expensive experiments.
    """)
    return


@app.cell
def _(np, plt):
    headline_labels = ["Randomized AMA", "CA-AMA"]
    headline_paper = np.array([3.1363, 3.6205])
    headline_observed = np.array([3.053014, 3.735941])
    headline_x = np.arange(2)
    headline_width = 0.34
    headline_fig, headline_ax = plt.subplots(figsize=(7.2, 3.5))
    headline_ax.bar(
        headline_x - headline_width / 2,
        headline_paper,
        headline_width,
        label="Paper",
        color="#687386",
    )
    headline_ax.bar(
        headline_x + headline_width / 2,
        headline_observed,
        headline_width,
        label="Observed, five seeds",
        color=["#5975A4", "#D65F5F"],
    )
    headline_ax.set_xticks(headline_x, headline_labels)
    headline_ax.set_ylabel("Expected revenue")
    headline_ax.set_title("Claim 4 headline comparison")
    headline_ax.legend(frameon=False)
    headline_ax.grid(axis="y", alpha=0.25)
    headline_fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The mechanism in one equation

    CA-AMA adds a correlation-aware term to the ordinary affine-maximizer
    payment:

    \[
    p_i(v)=p_i^{AMA}(v)+p_i^{Cor}(v_{-i}).
    \]

    Because \(p_i^{Cor}\) sees only rival values, it is identical under a
    truthful report and an own-value misreport. It cancels from the utility
    difference. The reproduction checks that identity exactly, then tests
    800 misreports on feasible multi-item menus. An invalid own-bid-dependent
    payment is the negative control and produces a profitable deviation.
    """)
    return


@app.cell
def _(mo):
    scope_delta = mo.ui.slider(
        start=0.01,
        stop=0.5,
        step=0.01,
        value=0.1,
        label="Requested approximation factor δ",
    )
    scope_delta
    return (scope_delta,)


@app.cell
def _(mo, np, scope_delta):
    selected_delta = float(scope_delta.value)
    selected_eta = np.exp(-2.0 / selected_delta)
    selected_n_ge_2_ratio = (
        selected_delta / 2.0
        + selected_eta * (1.0 - selected_eta) * selected_delta / 2.0
    )
    mo.md(
        rf"""
        ## Why the literal “any number of bidders” scope is false

        For \(n\ge2\), the paper's construction gives an upper-bound ratio of
        **{selected_n_ge_2_ratio:.6g}**, below the requested
        \(\delta={selected_delta:.2f}\).

        For \(n=1\), every DSIC/IR single-item mechanism is a mixture of
        posted prices, and deterministic AMA implements the best posted price.
        Its positive-revenue ratio is therefore **1**, not at most
        \(\delta<1\).

        Claims 1 and 2 are marked **FALSIFIED** under that literal source
        quantifier. Their intended \(n\ge2\) constructions remain supported.
        """
    )
    return


@app.cell
def _(np, plt):
    seed_ids = np.array([1, 7, 19, 41, 73])
    seed_revenues = np.array(
        [3.732785, 3.731810, 3.744466, 3.734596, 3.736045]
    )
    seed_regrets = np.array(
        [0.002729, 0.002902, 0.002792, 0.002799, 0.002842]
    )
    seed_fig, (seed_left, seed_right) = plt.subplots(1, 2, figsize=(8.4, 3.3))
    seed_left.plot(seed_ids, seed_revenues, "o-", color="#D65F5F")
    seed_left.axhline(3.6205, color="#687386", linestyle="--")
    seed_left.set_title("CA-AMA revenue")
    seed_left.set_xlabel("Training seed")
    seed_left.grid(alpha=0.25)
    seed_right.plot(seed_ids, seed_regrets, "o-", color="#55A868")
    seed_right.axhline(0.0031, color="#687386", linestyle="--")
    seed_right.set_title("Ex-post IR regret")
    seed_right.set_xlabel("Training seed")
    seed_right.grid(alpha=0.25)
    seed_fig.suptitle("Claim 4 stability across five independent seeds")
    seed_fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Claim 4 is still **BLOCKED**, despite passing the numerical tolerances.
    The public release has no matching learned 3 × 10, 2048-menu AMenuNet
    checkpoint and publishes a contradictory 10 × 3 command. The
    reproduction uses a validated separable reserve core and held-out
    payment scaling, which are material substitutions.

    ## Claim 5: what the exact bound can and cannot say

    The paper's Bernoulli-mixture distribution implies an exact expected
    welfare bound

    \[
    5\left[\frac35\frac{21}{40}
    +\frac25\frac{49}{96}\right]
    =\frac{623}{240}=2.59583.
    \]

    The reported CA revenue 1.9359 is below welfare plus reported regret
    0.0052, and reported ex-post revenue 1.8553 is below welfare. The
    values are feasible, so the mandatory falsification route finds no
    counterexample. The faithful CPU optimization remains unavailable;
    Claim 5 is **BLOCKED**.
    """)
    return


@app.cell
def _(mo):
    claim_rows = [
        ["1", "FALSIFIED", "HIGH", "Literal n=1 scope"],
        ["2", "FALSIFIED", "HIGH", "Correlated separation at n=1"],
        ["3", "VERIFIED", "HIGH", "Exact rival-only cancellation"],
        ["4", "BLOCKED", "MEDIUM", "Exact 3×10 learned core unavailable"],
        ["5", "BLOCKED", "LOW", "Four routes; full CPU optimization unavailable"],
    ]
    mo.md(
        "## Final evidence status\n\n"
        + mo.as_html(
            mo.ui.table(
                claim_rows,
                headers=["Claim", "Verdict", "Confidence", "Basis"],
            )
        ).text
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reproduce the committed evidence

    The formal command is fixed across scientific branches:

    ```bash
    uv run --frozen python repro/src/run_caama.py && \
    uv run --frozen python -m pytest -q repro/tests
    ```

    The winning evidence SHA is
    `bf4cc9371feea65edf71ad1dc998ed88de23b7a7`. It produced 89
    SHA-256-manifested artifacts and passed 25 tests on local CPU.

    This notebook is a tutorial surface. Formal verdicts come from the
    fail-closed claim verifiers and independent checker outputs, not from
    changing interactive controls here.
    """)
    return


if __name__ == "__main__":
    app.run()
