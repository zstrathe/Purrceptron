# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""VisitIdentifier and FeedingEventLogger: names and log rows per visit."""
import sqlite3
from datetime import datetime

import pytest

from deploy.cat_classifier import FeedingEventLogger, VisitIdentifier, VoteWindow, visit_record

CATS = ["Cat1", "Cat2", "Cat3"]
C1, C2, C3 = 0, 1, 2
REJECT = None


@pytest.fixture
def db(tmp_path):
    return tmp_path / "log.db"


def rows(db):
    with sqlite3.connect(db) as c:
        out = c.execute("SELECT cat_name, start_time, end_time, frame_count, in_progress "
                        "FROM classifier_events ORDER BY id").fetchall()
    ts = lambda v: datetime.fromisoformat(v).timestamp() if v else None
    return [(n, ts(s), ts(e), fc, ip) for n, s, e, fc, ip in out]


def make(db, **kw):
    return VisitIdentifier(CATS, FeedingEventLogger(db), window_size=10,
                           confidence_threshold=0.5, **kw)


def feed(ident, vid, seq, t0, dt=1.0, conf=0.9):
    """Crops of one visit, one per second from t0. REJECT = rejected crop."""
    t = t0
    for k in seq:
        if k is REJECT:
            ident.on_crop(vid, t, 0, 0.0, energy=0.0, rejected=True, mono=t)
        else:
            ident.on_crop(vid, t, k, conf, energy=-9.0, mono=t)
        t += dt
    return t


def rec(ident, vid):
    return visit_record(ident.status(), vid)


# ── naming and Unknown ────────────────────────────────────────────────────

def test_named_at_the_sixth_agreeing_crop(db):
    ident = make(db)
    feed(ident, 100.0, [C1] * 5, 101)
    assert rows(db) == [] and not rec(ident, 100.0)["unknown"]
    feed(ident, 100.0, [C1], 106)
    (r,) = rows(db)
    assert r[0] == "Cat1" and r[1] == pytest.approx(100.0) and r[4] == 1
    assert rec(ident, 100.0)["visit_cat"] == "Cat1"


def test_sustained_rejection_is_unknown(db):
    ident = make(db)
    feed(ident, 100.0, [REJECT] * 4, 101)
    assert not rec(ident, 100.0)["unknown"]
    feed(ident, 100.0, [REJECT], 105)
    r = rec(ident, 100.0)
    assert r["unknown"] and r["ood_rejected"] and r["prediction"] is None


def test_no_winner_after_a_full_window_is_unknown(db):
    ident = make(db)
    feed(ident, 100.0, [C1, C2] * 4 + [C1], 101)
    assert not rec(ident, 100.0)["unknown"]
    feed(ident, 100.0, [C2], 110)
    assert rec(ident, 100.0)["unknown"]


def test_unknown_cleared_by_a_later_name(db):
    ident = make(db)
    t = feed(ident, 100.0, [REJECT] * 5 + [C1], 101)
    assert rec(ident, 100.0)["unknown"]                 # sticks while one crop is accepted
    feed(ident, 100.0, [C1] * 5, t)
    r = rec(ident, 100.0)
    assert r["visit_cat"] == "Cat1" and not r["unknown"]


def test_named_visit_keeps_its_name_when_crops_are_rejected(db):
    ident = make(db)
    t = feed(ident, 100.0, [C1] * 6, 101)
    feed(ident, 100.0, [REJECT] * 5, t)
    r = rec(ident, 100.0)
    assert r["visit_cat"] == "Cat1" and not r["unknown"] and r["ood_rejected"]


def test_departure_closes_the_row_at_last_seen(db):
    ident = make(db)
    feed(ident, 100.0, [C1] * 8, 101)
    ident.on_departed(100.0, 109.5)
    (r,) = rows(db)
    assert r[2] == pytest.approx(109.5) and r[4] == 0 and r[3] == 3   # first vote + 2 later
    assert ident.status()["visits"] == []


def test_departure_of_unknown_visit_is_ignored(db):
    ident = make(db)
    ident.on_departed(5.0, 6.0)
    assert rows(db) == []


# ── several visits ────────────────────────────────────────────────────────

def test_two_interleaved_visits_get_two_rows(db):
    ident = make(db)
    for i in range(6):
        ident.on_crop(100.0, 101 + i, C1, 0.9, mono=101 + i)
        ident.on_crop(100.5, 101.5 + i, C2, 0.9, mono=101.5 + i)
    ident.on_departed(100.0, 110)
    ident.on_departed(100.5, 111)
    assert [(r[0], r[4]) for r in rows(db)] == [("Cat1", 0), ("Cat2", 0)]


def test_name_held_by_an_animal_in_view_is_not_given_twice(db):
    ident = make(db)
    t = feed(ident, 100.0, [C2] * 6, 101)                # holder named Cat2
    for i in range(6):                                   # both seen at the same time
        ident.on_crop(100.0, t + i, C2, 0.9, mono=t + i)
        ident.on_crop(104.0, t + i + 0.5, C2, 0.9, mono=t + i + 0.5)
    r = rec(ident, 104.0)
    assert r["name_taken"] == "Cat2" and r["unknown"] and r["visit_cat"] is None
    assert len(rows(db)) == 1
    # The holder leaves; the second visit's vote still says Cat2: it continues
    # the holder's row (within the rejoin time) instead of opening a second one.
    ident.on_departed(100.0, t + 6)
    feed(ident, 104.0, [C2], t + 7)
    r = rec(ident, 104.0)
    assert r["visit_cat"] == "Cat2" and r["name_taken"] is None
    assert len(rows(db)) == 1 and rows(db)[0][4] == 1


def test_lost_and_found_takes_over_the_row(db):
    ident = make(db)
    t = feed(ident, 100.0, [C1] * 6, 101)                # last seen at 106
    feed(ident, 107.0, [C1] * 6, 108)                    # tracker found it again, no departure yet
    assert len(rows(db)) == 1 and rec(ident, 100.0) is None
    assert rec(ident, 107.0)["visit_cat"] == "Cat1"
    pairs = ident.status()["confirmed_visits"]
    assert [100.0, "Cat1"] in pairs and [107.0, "Cat1"] in pairs


# ── joining split visits ──────────────────────────────────────────────────

@pytest.mark.parametrize("gap,joined", [(1.0, True), (4.0, True), (10.0, False)])
def test_same_animal_back_soon_continues_its_row(db, gap, joined):
    ident = make(db)
    feed(ident, 100.0, [C1] * 6, 101)
    ident.on_departed(100.0, 106.0)
    start = 106.0 + gap
    feed(ident, start, [C1] * 6, start + 1)
    ident.on_departed(start, start + 7)
    rs = rows(db)
    if joined:
        assert len(rs) == 1 and rs[0][1] == pytest.approx(100.0) and rs[0][2] == pytest.approx(start + 7)
    else:
        assert len(rs) == 2


def test_other_animal_back_soon_gets_its_own_row(db):
    ident = make(db)
    feed(ident, 100.0, [C1] * 6, 101)
    ident.on_departed(100.0, 106.0)
    feed(ident, 107.0, [C2] * 6, 108)
    ident.on_departed(107.0, 114.0)
    rs = rows(db)
    assert [r[0] for r in rs] == ["Cat1", "Cat2"] and rs[0][2] == pytest.approx(106.0)


def test_rejoin_off(db):
    ident = make(db, rejoin_seconds=0)
    feed(ident, 100.0, [C1] * 6, 101)
    ident.on_departed(100.0, 106.0)
    feed(ident, 107.0, [C1] * 6, 108)
    assert len(rows(db)) == 2


# ── a row never holds two animals ─────────────────────────────────────────

def test_two_odd_crops_change_nothing(db):
    ident = make(db)
    t = feed(ident, 100.0, [C1] * 6, 101)
    feed(ident, 100.0, [C2, C2, C1, C1], t)
    ident.on_departed(100.0, 120)
    (r,) = rows(db)
    assert r[0] == "Cat1" and r[3] == 3          # the confirming vote + 2 agreeing; Cat2 crops not counted


def test_another_animal_takes_the_place(db):
    ident = make(db)
    t = feed(ident, 100.0, [C1] * 10, 101)           # 101..110
    t = feed(ident, 100.0, [C2] * 6, t)              # 111..116: Cat2 confirmed at 116
    r = rec(ident, 100.0)
    assert r["visit_cat"] == "Cat2"
    ident.on_departed(100.0, 120)
    a, b = rows(db)
    assert a[0] == "Cat1" and a[2] == pytest.approx(111.0)    # ends at Cat2's first vote
    assert b[0] == "Cat2" and b[1] == pytest.approx(111.0) and b[2] == pytest.approx(120.0)


def test_flip_back_never_puts_two_animals_in_a_row(db):
    ident = make(db)
    t = feed(ident, 100.0, [C1] * 10, 101)                    # 101..110
    t = feed(ident, 100.0, [C2] * 6, t)                       # 111..116
    feed(ident, 100.0, [C1] * 6, t)                           # 117..122
    ident.on_departed(100.0, 130)
    rs = rows(db)
    # Cat2 held the place 6 s, longer than the 5 s join window: three rows,
    # back to back, each with one animal.
    assert [r[0] for r in rs] == ["Cat1", "Cat2", "Cat1"]
    assert rs[0][2] == pytest.approx(111) and rs[1][1] == pytest.approx(111)
    assert rs[1][2] == pytest.approx(117) and rs[2][1] == pytest.approx(117)
    assert rs[2][2] == pytest.approx(130)


def test_short_flip_back_rejoins_the_first_row(db):
    ident = make(db, rejoin_seconds=10)
    t = feed(ident, 100.0, [C1] * 10, 101)
    t = feed(ident, 100.0, [C2] * 6, t)
    feed(ident, 100.0, [C1] * 6, t)
    ident.on_departed(100.0, 130)
    rs = rows(db)
    assert [r[0] for r in rs] == ["Cat1", "Cat2"] and rs[0][2] == pytest.approx(130)


def test_two_animals_swap_places(db):
    ident = make(db)
    for i in range(10):                                       # left = Cat1, right = Cat2
        ident.on_crop(100.0, 101 + i, C1, 0.9, mono=101 + i)
        ident.on_crop(100.5, 101.5 + i, C2, 0.9, mono=101.5 + i)
    for i in range(10, 17):                                   # they swap
        ident.on_crop(100.0, 101 + i, C2, 0.9, mono=101 + i)
        ident.on_crop(100.5, 101.5 + i, C1, 0.9, mono=101.5 + i)
    assert rec(ident, 100.0)["visit_cat"] == "Cat2" and rec(ident, 100.5)["visit_cat"] == "Cat1"
    ident.on_departed(100.0, 120)
    ident.on_departed(100.5, 121)
    rs = rows(db)
    assert len(rs) == 2 and sorted(r[0] for r in rs) == ["Cat1", "Cat2"]
    assert all(r[1] < 101 for r in rs)                        # neither row was split


def test_change_to_a_name_still_voted_elsewhere_is_unidentified(db):
    ident = make(db)
    for i in range(6):
        ident.on_crop(100.0, 101 + i, C1, 0.9, mono=101 + i)
    for i in range(6):                                        # another visit holds Cat2
        ident.on_crop(103.0, 104 + i, C2, 0.9, mono=104 + i)
    for i in range(6, 12):
        ident.on_crop(100.0, 101 + i, C2, 0.9, mono=101 + i)  # first visit drifts to Cat2
        ident.on_crop(103.0, 104 + i, C2, 0.9, mono=104 + i)
    r = rec(ident, 100.0)
    assert r["visit_cat"] is None and r["name_taken"] == "Cat2" and r["unknown"]
    cat1 = [x for x in rows(db) if x[0] == "Cat1"][0]
    assert cat1[4] == 0                                       # Cat1's row was closed


# ── staleness, shutdown ─────────────────────────────────────────

def test_stale_visit_is_ended_at_last_seen(db):
    ident = make(db)
    feed(ident, 100.0, [C1] * 6, 101)
    ident.expire_stale(mono=106 + 30)
    assert rows(db)[0][4] == 1
    ident.expire_stale(mono=106 + 61)
    assert rows(db)[0][4] == 0 and rows(db)[0][2] == pytest.approx(106.0)


def test_abandon_all_drops_new_rows_and_restores_joined_ones(db):
    ident = make(db)
    feed(ident, 100.0, [C1] * 6, 101)
    ident.on_departed(100.0, 106.0)
    feed(ident, 107.0, [C1] * 6, 108)                         # joined into the first row
    feed(ident, 107.5, [C2] * 6, 108.5)                       # new row
    ident.abandon_all()
    rs = rows(db)
    assert rs == [("Cat1", pytest.approx(100.0), pytest.approx(106.0), 1, 0)]


def test_logger_startup_keeps_continued_rows(db):
    lg = FeedingEventLogger(db)
    a = lg.open("Cat1", 0.9, 100.0); lg.close(a, 110.0)
    lg.reopen_recent("Cat1", 105.0)                           # continued, then a crash
    lg.open("Cat2", 0.9, 120.0)                               # new, then a crash
    FeedingEventLogger(db)
    assert rows(db) == [("Cat1", pytest.approx(100.0), pytest.approx(110.0), 1, 0)]


def test_logger_reopen_restores_average_and_count(db):
    lg = FeedingEventLogger(db)
    a = lg.open("Cat1", 0.8, 100.0); lg.vote(a, 1.0); lg.close(a, 110.0)
    b = lg.reopen_recent("Cat1", 108.0)
    assert b == a
    lg.vote(b, 0.6); lg.close(b, 120.0)
    with sqlite3.connect(db) as c:
        avg, fc = c.execute("SELECT confidence_avg, frame_count FROM classifier_events").fetchone()
    assert fc == 3 and avg == pytest.approx(0.8)


def test_status_lists_visits(db):
    ident = make(db)
    feed(ident, 100.0, [C1] * 6, 101)
    st = ident.status()
    assert [v["visit_id"] for v in st["visits"]] == [100.0] and st["visits"][0]["confirmed"]
    assert st["confirmed_visits"] == [[100.0, "Cat1"]]


def test_vote_window_first_seen_ignores_a_stray_early_vote():
    w = VoteWindow(10, 0.5)
    for i, k in enumerate([C2, C1, C1, C1, C1, C2, C2, C2, C2, C2]):
        w.add(k, 0.9, 100 + i)
    assert w.first_seen_of(C2) == 105
