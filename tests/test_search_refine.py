"""src/race_level.py grid_neighbors / refine_search: the local refinement after random search."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from race_level import grid_neighbors, refine_search, sample_grid  # noqa: E402

GRID = dict(depth=[1, 2, 3], lam=[0.25, 1, 5])


def test_neighbors_step_one_value_at_a_time():
    n = grid_neighbors(dict(depth=1, lam=1), GRID)
    assert dict(depth=2, lam=1) in n and dict(depth=1, lam=0.25) in n and dict(depth=1, lam=5) in n
    assert len(n) == 3                                   # depth 0 does not exist


def test_refine_finds_the_better_neighbour_and_keeps_order():
    true = lambda p: abs(p["depth"] - 3) + abs(p["lam"] - 0.25)     # best at depth 3, lam 0.25
    stage1 = sorted([(true(p), p) for p in (dict(depth=2, lam=1), dict(depth=1, lam=5))], key=lambda t: t[0])
    out = refine_search(stage1, GRID, true, lambda t: t[0], top=2)
    assert out[0][1] == dict(depth=3, lam=0.25)          # the hill-climb reaches the optimum
    assert [s for s, _ in out] == sorted(s for s, _ in out)
    assert len({tuple(sorted(p.items())) for _, p in out}) == len(out)  # nothing scored twice


def test_refine_climbs_several_steps():
    # best is 2 steps from every stage-1 point: one pass could not reach it, the hill-climb must
    g = dict(depth=[1, 2, 3, 4, 5], lam=[0, 1, 2, 3, 4])
    true = lambda p: abs(p["depth"] - 5) + abs(p["lam"] - 0)
    stage1 = [(true(p), p) for p in (dict(depth=3, lam=2),)]
    out = refine_search(stage1, g, true, lambda t: t[0], top=1)
    assert out[0][1] == dict(depth=5, lam=0)


def test_sample_grid_is_distinct_reproducible_and_capped():
    g = dict(a=[1, 2, 3], b=[1, 2])
    s1, s2 = sample_grid(g, 4, seed=7), sample_grid(g, 4, seed=7)
    assert s1 == s2 and len({tuple(sorted(p.items())) for p in s1}) == 4
    assert len(sample_grid(g, 99)) == 6                  # never more than the grid holds
