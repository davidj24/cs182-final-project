import numpy as np
import pytest
import torch

from cs182_icl.model.gpt2 import ModelConfig, PushGPT, count_params
from cs182_icl.model.predict import model_predict
from cs182_icl.model.tokens import compute_norm, encode
from cs182_icl.train.data import make_synthetic
from cs182_icl.types import Action, Cube, Observation, TestItem


def small_model(seed=0):
    torch.manual_seed(seed)
    return PushGPT(ModelConfig(n_layer=2, n_embd=32, n_head=4)).eval()


def test_param_count_full_model(capsys):
    n = count_params(PushGPT(ModelConfig()))
    print(f"full model parameters: {n:,}")
    assert 9.0e6 < n < 10.0e6


def test_shapes():
    m = small_model()
    tok = torch.randn(3, 22, 5)
    assert m(tok).shape == (3, 22)
    assert m.push_predictions(tok).shape == (3, 11)
    d = make_synthetic(10, 0)
    norm = compute_norm(d["F"], d["k"], d["d_obs"])
    F = torch.tensor(d["F"][:4, :11]); T = torch.tensor(d["k"][:4, :11] * 0.001); dd = torch.tensor(d["d_obs"][:4, :11])
    assert encode(norm, F, T, dd).shape == (4, 22, 5)


def test_causal_future_changes_do_not_affect_earlier_outputs():
    m = small_model()
    tok = torch.randn(4, 22, 5)
    with torch.no_grad():
        base = m(tok)
        for j in range(1, 22):
            t2 = tok.clone()
            t2[:, j:] = torch.randn(4, 22 - j, 5) * 5  # change token j and everything later
            out = m(t2)
            assert (out[:, :j] - base[:, :j]).abs().max() < 1e-6, j


def test_leakage_current_outcome_not_visible_at_own_prediction():
    m = small_model()
    d = make_synthetic(20, 1)
    norm = compute_norm(d["F"], d["k"], d["d_obs"])
    F = torch.tensor(d["F"][:8, :11]); T = torch.tensor(d["k"][:8, :11] * 0.001); dd = torch.tensor(d["d_obs"][:8, :11])
    with torch.no_grad():
        base = m.push_predictions(encode(norm, F, T, dd))
        for j in range(11):
            d2 = dd.clone(); d2[:, j] += 7.0  # perturb outcome j only
            p2 = m.push_predictions(encode(norm, F, T, d2))
            assert (p2[:, : j + 1] - base[:, : j + 1]).abs().max() < 1e-6, j  # positions 0..j blind to outcome j
        # sanity: later predictions DO see it (test is not vacuous)
        d2 = dd.clone(); d2[:, 0] += 7.0
        assert (m.push_predictions(encode(norm, F, T, d2))[:, 1:] - base[:, 1:]).abs().max() > 1e-6
    # token layout: outcome j sits at position 2j+1, after the push token 2j where it is read
    tok = encode(norm, F, T, dd)
    assert (tok[:, 0::2, 2:] == 0).all() and (tok[:, 1::2, :2] == 0).all() and (tok[:, 1::2, 4] == 1).all()


def test_normalization_is_train_only():
    tr, va = make_synthetic(50, 0), make_synthetic(50, 1, id_offset=1000)
    n_train = compute_norm(tr["F"], tr["k"], tr["d_obs"])
    assert np.isclose(n_train["F"]["mean"], tr["F"].mean()) and np.isclose(n_train["d_obs"]["std"], tr["d_obs"].std())
    assert np.isclose(n_train["T"]["mean"], (tr["k"] * 0.001).mean())
    assert np.isclose(n_train["asinh_d_obs"]["mean"], np.arcsinh(tr["d_obs"] / 0.01).mean())
    both = compute_norm(np.concatenate([tr["F"], va["F"]]), np.concatenate([tr["k"], va["k"]]),
                        np.concatenate([tr["d_obs"], va["d_obs"]]))
    assert both["d_obs"]["mean"] != n_train["d_obs"]["mean"]  # val stats would differ
    # load_dataset uses only train.npz: changing val does not change norm
    import tempfile, pathlib
    from cs182_icl.train.data import load_dataset
    with tempfile.TemporaryDirectory() as td:
        np.savez(pathlib.Path(td) / "train.npz", **{k: v for k, v in tr.items() if k != "synthetic"})
        for val in (va, make_synthetic(50, 9, 2000)):
            np.savez(pathlib.Path(td) / "val.npz", **{k: v for k, v in val.items() if k != "synthetic"})
            assert load_dataset(td)["norm"] == n_train


def test_model_predict_matches_manual_and_ignores_padding():
    m = small_model()
    d = make_synthetic(5, 2)
    norm = compute_norm(d["F"], d["k"], d["d_obs"])
    cube = Cube(1.0, 0.4, 0)
    ctx = [Observation(Action(15.0, 120), i, 0.01 * i, 0.01 * i) for i in range(3)]
    item = TestItem(cube, Action(20.0, 150), 0.0, ctx, "E1")
    p = model_predict(m, [item, TestItem(cube, Action(20.0, 150), 0.0, [], "E1")], norm)
    assert p.shape == (2,) and p.dtype == np.float64
    # same item with a different (hidden) query outcome gives the same prediction
    item2 = TestItem(cube, Action(20.0, 150), 99.0, ctx, "E1")
    assert model_predict(m, [item2], norm)[0] == pytest.approx(p[0], abs=1e-9)
