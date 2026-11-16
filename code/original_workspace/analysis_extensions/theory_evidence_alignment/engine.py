"""Exact full-risk sufficient moments for nested models and aligned path scores."""
from pathlib import Path
import os
import sys
import json
import hashlib
import shutil
import time

sys.dont_write_bytecode = True
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "expanded_sample_reassessment_20261102"
PORT = ROOT / "portfolio_A_theory_driven_exploration_20261103"
ANCHOR = ROOT / "anchor_amplification_exploration_20261106"
PAPER = ROOT / "knowledge_architecture_working_paper_revised_20261107"
sys.path.insert(0, str(BASE / "08_reproducibility"))

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.stats import t
from panel_api import Panel

OUTCOMES = ["total", "specialist", "no_specialist", "signed"]
ARCH = ["Gap", "Distance", "Distributed", "Shared"]
EXTRA = ["LogBreadth", "LogProductivity"]
EXPECTED = {
    "parent": (602919, 102614232, [9065, 3386, 5679, 9065]),
    "direct": (597077, 70422255, [7188, 2711, 4477, 7188]),
}
FOCAL = "Distributed_Home"


def note(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def freeze():
    folder = HERE / "frozen_table3"
    if (folder / "manifest.json").exists():
        for record in read(folder / "manifest.json"):
            assert digest(ROOT / record["original"]) == record["sha256"]
            assert digest(folder / record["snapshot"]) == record["sha256"]
        return
    crosswalk = pd.read_csv(PAPER / "evidence/table_crosswalk.csv")
    cells = crosswalk.query("final_table == 'Table 3' and statistic in ['b', 'se', 'p']")
    records = []
    def snapshot(path, relative):
        dest = folder / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        records.append(dict(original=str(path.relative_to(ROOT)), snapshot=str(relative), sha256=digest(path)))
    for mp, (cores, pairs, events) in EXPECTED.items():
        path = ANCHOR / "00_baseline_frozen/replication" / mp
        recon = pd.read_csv(path / "reconciliation.csv")
        assert len(recon) == 20 and recon.passed.all()
        assert (recon.N == pairs * 11).all()
        sample = read(path / "sample.json")
        assert (sample["cores"], sample["pairs"], sample["events"]) == (cores, pairs, events)
        for k, out in enumerate(OUTCOMES):
            f = path / (out + ".json")
            old = read(f)
            assert old["meta"]["nonzero_cells"] == events[k]
            rows = cells[cells.source_file.map(lambda v: Path(str(v)) == f)]
            assert len(rows) == 30, (mp, out, len(rows))
            for row in rows.itertuples():
                term = row.source_specification_id
                pointer = str(row.source_pointer).strip("/").split("/")
                obj = old
                for key in pointer:
                    obj = obj[int(key)] if isinstance(obj, list) else obj[key]
                value = obj[row.statistic] if isinstance(obj, dict) else obj
                assert np.isclose(value, float(row.raw_value), rtol=2e-12, atol=1e-16), (term, row.statistic)
            snapshot(f, Path(mp) / f.name)
        for name in ["joint_covariances.npz", "joint_metadata.json", "sample.json", "events.parquet", "reconciliation.csv"]:
            snapshot(path / name, Path(mp) / name)
    for path in [
        PAPER / "working_paper_revised.pdf", PAPER / "tables/Table3.tex",
        PAPER / "sections/design.tex", PAPER / "sections/setting.tex",
        BASE / "08_reproducibility/panel_api.py", BASE / "08_reproducibility/streaming_fe.py",
        BASE / "08_reproducibility/inputs/field_month_proxies.csv",
        BASE / "08_reproducibility/inputs/subfield_mapping_versions.csv",
        BASE / "08_reproducibility/inputs/pre2014_domain_counts.csv",
        PORT / "measurement/R.npy", PORT / "code/constructs.py",
    ]:
        snapshot(path, Path("sources") / path.name)
    save(folder / "manifest.json", records)
    note("Table 3 verified and frozen", len(records), "files")


class Risk:
    def __init__(self, mapping):
        self.mapping = mapping
        panel = Panel("recent", "dyad", mapping, classes="C")
        self.order = np.argsort(panel.fc, kind="stable")
        inverse = np.argsort(self.order)
        self.fields = panel.js[self.order]
        self.paths = panel.paths.tolist()
        self.path = panel.fc[self.order]
        self.groups = [slice(*np.searchsorted(self.path, [k, k + 1])) for k in range(len(self.paths))]
        self.mask = panel.mask[:, self.order]
        self.I, self.J = self.mask.shape
        self.npairs = int(self.mask.sum())
        self.N = self.npairs * panel.T
        self.den = self.mask.sum(1)
        self.H = panel.ha
        self.A = panel.A[self.order]
        self.transform = helmert(panel.T).T
        self.ht = self.H @ self.transform
        self.at = self.A @ self.transform
        self.T = self.ht.shape[1]
        self.R = np.load(PORT / "measurement/R.npy")
        self.sub = panel.s.reset_index(drop=True).iloc[self.fields].reset_index(drop=True)
        self.raw = {
            "Gap": panel.G[:, self.order],
            "Distance": panel.D[:, self.order],
        }
        for name, filename in [("Distributed", "distributed_input"), ("Shared", "shared_input")]:
            self.raw[name] = np.load(PORT / "measurement" / (filename + ".npy"), mmap_mode="r")[panel.g][:, self.fields]
        counts = pd.read_csv(BASE / "08_reproducibility/inputs/pre2014_domain_counts.csv").set_index("subfield_id")
        counts = counts.classified_citations.reindex(panel.s.subfield_id).to_numpy()
        prevalence = counts @ self.R / counts.sum()
        hist = panel.hist
        weights = hist / hist.sum(1, keepdims=True).clip(1)
        self.raw["Applicability"] = np.log1p(weights @ (self.R[self.fields] / prevalence.clip(1e-12)).T)
        self.raw["Access"] = np.log1p(np.load(panel.u / "network/network.npy", mmap_mode="r")[panel.g][:, self.fields])
        self.raw["LogBreadth"] = np.broadcast_to(np.log1p((hist > 0).sum(1))[:, None], self.mask.shape)
        self.raw["LogProductivity"] = np.broadcast_to(np.log1p(panel.mi.n_pre.to_numpy())[:, None], self.mask.shape)
        self.cores = panel.g
        self.z, self.scales = {}, {}
        for name, matrix in self.raw.items():
            valid = matrix[self.mask].astype(float)
            mu, sd = float(valid.mean()), float(valid.std())
            assert sd > 0
            self.scales[name] = {"mean": mu, "sd": sd}
            self.z[name] = (matrix.astype(float) - mu) / sd
        ref = read(HERE / "frozen_table3" / mapping / "sample.json")
        for name, moments in ref["standardization"].items():
            assert all(abs(self.scales[name][k] - v) < 1e-12 for k, v in moments.items())
        ref_a = read(PORT / "results/dyad/recent" / mapping / "requirements_C/scaling.json")
        assert all(abs(self.scales["Applicability"][k] - ref_a["Applicability"][k]) < 1e-12 for k in ["mean", "sd"])
        w = panel.ww[panel.ww.mo == panel.ww.first_mo].copy()
        w["specialist"] = w.current_experts > w.current_core_experts
        e = w.groupby(["i", "j0", "mo"], sort=True).agg(specialist=("specialist", "max"), papers=("work_id", "size")).reset_index()
        expected_events = pd.read_parquet(HERE / "frozen_table3" / mapping / "events.parquet")
        pd.testing.assert_frame_equal(e[expected_events.columns], expected_events, check_dtype=False)
        self.ei = e.i.to_numpy()
        self.ej = inverse[e.j0.to_numpy()]
        self.et = e.mo.to_numpy() - 786
        specialist = e.specialist.to_numpy(float)
        self.v = np.column_stack([specialist, 1 - specialist])
        self.counts = [len(e), int(specialist.sum()), int(len(e) - specialist.sum()), len(e)]
        assert (self.I, self.npairs, self.counts) == EXPECTED[mapping]
        assert self.mask[self.ei, self.ej].all()
        self.event_y = self.transform[self.et, :, None] * self.v[:, None, :]
        self.row_y = np.zeros((self.I, self.T, 2))
        np.add.at(self.row_y, self.ei, self.event_y)
        self.row_y /= self.den[:, None, None]
        panel.c.close()
        note("Risk loaded", mapping, self.I, self.J, self.npairs, self.counts)

    def blocks(self, chunk=3072):
        for a in range(0, self.I, chunk):
            yield a, min(a + chunk, self.I)

    def add_squared(self, names):
        for name in names:
            sq = self.z[name] ** 2
            valid = sq[self.mask]
            self.z[name + "Sq"] = (sq - valid.mean()) / valid.std()

    def project_destinations(self):
        s = np.diag(self.mask.sum(0).astype(float))
        rhs = np.zeros((self.J, self.T, 2))
        np.add.at(rhs, self.ej, self.event_y)
        for a, b in self.blocks():
            m = self.mask[a:b].astype(float)
            s -= m.T @ (m / self.den[a:b, None])
            rhs -= (m.T @ self.row_y[a:b].reshape(b-a, -1)).reshape(rhs.shape)
        ev, u = np.linalg.eigh(s)
        use = ev > ev.max() * 1e-12
        assert int(use.sum()) == self.J - 1
        self.sinv = (u[:, use] / ev[use]) @ u[:, use].T
        self.gamma_y = (self.sinv @ rhs.reshape(self.J, -1)).reshape(rhs.shape)
        raw = self.v.T @ self.v
        yy = raw * (self.T / (self.T + 1))
        yy -= np.einsum("i,itm,itn->mn", self.den, self.row_y, self.row_y, optimize=True)
        self.yy = yy - rhs.reshape(-1, 2).T @ self.gamma_y.reshape(-1, 2)


def moments(risk, columns, mods, values, destination):
    """Partial out pair, core-month, destination-month once for a union of models."""
    if destination.exists():
        with np.load(destination, allow_pickle=False) as stored:
            assert stored["columns"].tolist() == columns and stored["mods"].tolist() == mods
            return {key: stored[key] for key in stored.files}
    if not hasattr(risk, "sinv"):
        risk.project_destinations()
    J, T, K, L = risk.J, risk.T, len(columns), len(mods)
    G = len(risk.paths)
    sources = [name.rsplit("_", 1)[-1] for name in columns]
    names = [name.rsplit("_", 1)[0] for name in columns]
    to_dest = [k for k, source in enumerate(sources) if source == "Destination"]
    to_home = [k for k, source in enumerate(sources) if source == "Home"]
    assert to_dest + to_home == list(range(K)), "source columns must be contiguous"
    cut = len(to_dest)
    row_x = np.empty((risk.I, T, K))
    row_z = np.empty((risk.I, L))
    rhs_x = np.zeros((J, T, K))
    rhs_z = np.zeros((J, L))

    def build(a, b):
        z = np.stack([values[name][a:b] for name in mods], axis=-1)
        xd = np.stack([values[names[k]][a:b] for k in to_dest], axis=-1)
        xh = np.stack([values[names[k]][a:b] for k in to_home], axis=-1)
        x = np.empty((b-a, J, T, K))
        np.multiply(xd[:, :, None, :], risk.at[None, :, :, None], out=x[:, :, :, :cut])
        np.multiply(xh[:, :, None, :], risk.ht[a:b, None, :, None], out=x[:, :, :, cut:])
        return x, z

    for pass_index, (a, b) in enumerate(risk.blocks()):
        if pass_index % 48 == 0:
            note(destination.stem, "projection", a, "/", risk.I)
        m = risk.mask[a:b]
        x, z = build(a, b)
        x *= m[:, :, None, None]
        z *= m[:, :, None]
        rx = row_x[a:b]
        rz = row_z[a:b]
        rx[:] = x.sum(1) / risk.den[a:b, None, None]
        rz[:] = z.sum(1) / risk.den[a:b, None]
        rhs_x += x.sum(0) - (m.T @ rx.reshape(b-a, -1)).reshape(J, T, K)
        rhs_z += z.sum(0) - m.T @ rz
    gamma_x = (risk.sinv @ rhs_x.reshape(J, -1)).reshape(J, T, K)
    gamma_z = risk.sinv @ rhs_z
    Q = np.zeros((G, K, K))
    xy = np.zeros((G, K, 2))
    Sz = np.zeros((G, L, L))
    Cx = np.zeros((G, K, L, T))
    zy = np.zeros((G, L, T, 2))
    yy = np.zeros((2, 2))
    for pass_index, (a, b) in enumerate(risk.blocks()):
        if pass_index % 48 == 0:
            note(destination.stem, "moments", a, "/", risk.I)
        m = risk.mask[a:b]
        x, z = build(a, b)
        row_adj = (m @ gamma_x.reshape(J, -1)).reshape(b-a, T, K) / risk.den[a:b, None, None]
        x -= gamma_x[None, :, :, :]
        x -= (row_x[a:b] - row_adj)[:, None, :, :]
        x *= m[:, :, None, None]
        z -= gamma_z[None, :, :]
        z -= (row_z[a:b] - (m @ gamma_z) / risk.den[a:b, None])[:, None, :]
        z *= m[:, :, None]
        y = np.broadcast_to(-risk.gamma_y, (b-a, J, T, 2)).copy()
        ry = risk.row_y[a:b] - (m @ risk.gamma_y.reshape(J, -1)).reshape(b-a, T, 2) / risk.den[a:b, None, None]
        y -= ry[:, None, :, :]
        left, right = np.searchsorted(risk.ei, [a, b])
        np.add.at(y, (risk.ei[left:right]-a, risk.ej[left:right]), risk.event_y[left:right])
        y *= m[:, :, None, None]
        for g, sl in enumerate(risk.groups):
            xf = np.ascontiguousarray(x[:, sl]).reshape(-1, T, K)
            zf = np.ascontiguousarray(z[:, sl]).reshape(-1, L)
            yf = np.ascontiguousarray(y[:, sl]).reshape(-1, T, 2)
            xflat, yflat = xf.reshape(-1, K), yf.reshape(-1, 2)
            Q[g] += xflat.T @ xflat
            xy[g] += xflat.T @ yflat
            Sz[g] += zf.T @ zf
            Cx[g] += (zf.T @ xf.reshape(-1, T*K)).reshape(L, T, K).transpose(2, 0, 1)
            zy[g] += (zf.T @ yf.reshape(-1, T*2)).reshape(L, T, 2)
            yy += yflat.T @ yflat
    assert np.allclose(yy, risk.yy, atol=1e-8, rtol=2e-10)
    result = dict(Q=Q, xy=xy, Sz=Sz, Cx=Cx, zy=zy, columns=np.array(columns), mods=np.array(mods))
    pending = destination.with_suffix(".tmp.npz")
    np.savez_compressed(pending, **result)
    pending.replace(destination)
    return result


def fit(risk, cache, columns, slopes):
    c = [cache["columns"].tolist().index(name) for name in columns]
    z = [cache["mods"].tolist().index(name) for name in slopes]
    K, L, T, G = len(c), len(z), risk.T, len(risk.paths)
    q = K + L*T
    group_Q, group_xy = np.zeros((G, q, q)), np.zeros((G, q, 2))
    for g in range(G):
        group_Q[g, :K, :K] = cache["Q"][g][np.ix_(c, c)]
        group_xy[g, :K] = cache["xy"][g, c]
        cross = cache["Cx"][g][np.ix_(c, z, np.arange(T))].reshape(K, -1)
        group_Q[g, :K, K:] = cross
        group_Q[g, K:, :K] = cross.T
        group_Q[g, K:, K:] = np.kron(cache["Sz"][g][np.ix_(z, z)], np.eye(T))
        group_xy[g, K:] = cache["zy"][g, z].reshape(-1, 2)
    Q = group_Q.sum(0)
    xy = group_xy.sum(0)
    d = np.sqrt(np.maximum(np.diag(Q), 1e-300))
    norm = Q / np.outer(d, d)
    ev = np.linalg.eigvalsh((norm + norm.T) / 2)
    rank = int(np.linalg.matrix_rank(norm))
    meta = dict(rank=rank, columns=q, condition_normalized=float(np.linalg.cond(norm)),
                min_normalized_eigenvalue=float(ev[0]), N=risk.N, cores=risk.I, pairs=risk.npairs,
                paths=risk.paths, terms=columns, slopes=slopes, events=risk.counts)
    if rank != q or ev[0] < 1e-12:
        return {**meta, "status": "rank_deficient_or_near_singular"}, None
    inv = np.linalg.inv(norm) / np.outer(d, d)
    B = inv @ xy
    scores = group_xy - group_Q @ B
    assert np.max(np.abs(scores.sum(0))) < 5e-8
    outcome = np.array([[1, 1, 0, 1], [1, 0, 1, -1]])
    beta = B[:K] @ outcome
    influence = (inv[:K] @ scores) @ outcome
    factor = (risk.N - 1) / (risk.N - risk.npairs - (risk.I + risk.J - 1)*T - q)
    adjusted = influence * np.sqrt(factor * G / (G-1))
    flat = adjusted.transpose(0, 2, 1).reshape(G, 4*K)
    cov = flat.T @ flat
    assert np.linalg.eigvalsh(cov)[0] >= -1e-10 * np.abs(cov).max()
    raw = risk.v.T @ risk.v
    sst = raw - np.outer(risk.v.sum(0), risk.v.sum(0)) / risk.N
    rss = risk.yy - B.T @ xy
    fe_only = risk.yy - xy[K:].T @ np.linalg.solve(Q[K:, K:], xy[K:])
    diagonal = lambda mat: np.diag(outcome.T @ mat @ outcome)
    meta.update(status="estimated", CR1=factor, df_inference=G-1,
                df_FE=risk.npairs + (risk.I + risk.J - 1 + L)*T,
                R2=(1-diagonal(rss)/diagonal(sst)).tolist(),
                within_R2=(1-diagonal(rss)/diagonal(fe_only)).tolist(),
                max_normal_equation_error=float(np.max(np.abs(scores.sum(0)))))
    result = dict(beta=beta, covariance=cov, influence=adjusted, raw_influence=influence,
                  Q_by_path=group_Q, xy_by_path=group_xy, inverse=inv)
    return meta, result


def terms(mods):
    return [name + "_Destination" for name in mods + EXTRA] + [name + "_Home" for name in mods]


def stats(b, scores, df=9):
    se = float(np.linalg.norm(scores))
    if not se or not np.isfinite(se):
        return dict(b=float(b), se=None, p=None, lo=None, hi=None, variance_valid=False)
    crit = t.ppf(.975, df)
    return dict(b=float(b), se=se, p=float(2*t.sf(abs(b/se), df)),
                lo=float(b-crit*se), hi=float(b+crit*se), variance_valid=True)


def verify_reproduction(mapping, meta, result):
    base = HERE / "frozen_table3" / mapping
    old = np.load(base / "joint_covariances.npz")
    order = [meta["terms"].index(n) for n in read(base / "joint_metadata.json")["regressors"]]
    ix = (np.arange(4)[:, None] * len(order) + np.array(order)[None, :]).ravel()
    be = np.max(np.abs(result["beta"][order].T - old["beta"]))
    ve = np.max(np.abs(result["covariance"][np.ix_(ix, ix)] - old["capability"]))
    assert be < 2e-11 and ve < 1e-13, (mapping, be, ve)
    note("Exact engine verified against frozen Table 3", mapping, be, ve)
    save(HERE / "diagnostics" / (mapping + "_reproduction.json"), dict(
        max_beta_error=float(be), max_joint_capability_covariance_error=float(ve), N=meta["N"],
        events=meta["events"], df_FE=meta["df_FE"], R2=meta["R2"], within_R2=meta["within_R2"],
    ))
