import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researcher_profile import Profile as P  # noqa: E402


@dataclass
class DomainGrade:
    """Оценка статьи по области в формате контракта taxonomyGrader (см. начало Profile.py)."""
    domain: str
    term_share: float
    term_depth_sum: float
    tool_depth_sum: float = 0.0


class FakeGrader:
    """Подмена taxonomyGrader: оценки заранее заданы по id статьи."""

    def __init__(self, grades):
        self.grades = grades

    def grade(self, paper):
        return self.grades[paper["id"]]


def make_profile(papers):
    profile = P.Profile("Иванов", 40, "a@b.c")
    profile.orcid = "0000-0000-0000-0001"
    for paper in papers:
        profile.upload_paper(paper)
    return profile


# top_n=None: тесты формул составляющих не зависят от domain_scores
CFG = dict(k1=0.5, k2=0.5, k3=0.5, k4=0.5, decay=1.0, current_year=2026, top_n=None)


def test_p_ex_formula():
    papers = [{"id": 1, "year": 2026, "coauthors": []}, {"id": 2, "year": 2023, "coauthors": []}]
    grader = FakeGrader({1: [DomainGrade("phys", 1.0, 0)], 2: [DomainGrade("phys", 1.0, 0)]})
    e = make_profile(papers).calculate_expertise(grader, P.ExpertiseConfig(**CFG))["phys"]
    assert e.p_ex == pytest.approx(math.exp(0) + math.exp(-1.0 * 3 / 3))


def test_domain_selection_95_percent():
    grades = [DomainGrade("a", 0.60, 1), DomainGrade("b", 0.36, 1), DomainGrade("c", 0.04, 1)]
    selected = P.Profile._select_domains(grades, 0.95)
    assert [g.domain for g in selected] == ["a", "b"]          # 0.60 + 0.36 = 0.96 >= 0.95, "c" не нужна


def test_p_t_sums_term_and_tool_depths():
    papers = [{"id": 1, "year": 2026, "coauthors": []}, {"id": 2, "year": 2026, "coauthors": []}]
    grader = FakeGrader({1: [DomainGrade("chem", 1.0, term_depth_sum=7, tool_depth_sum=3)],
                         2: [DomainGrade("chem", 1.0, term_depth_sum=5)]})
    e = make_profile(papers).calculate_expertise(grader, P.ExpertiseConfig(**CFG, tool_weight=0.5))["chem"]
    assert e.p_t == pytest.approx(7 + 0.5 * 3 + 5)


def test_p_team_counts_interdisciplinary_coauthored_papers_and_teams():
    papers = [
        {"id": 1, "year": 2026, "coauthors": ["X", "Y"]},   # phys 40% -> междисциплинарная, команда {X,Y}
        {"id": 2, "year": 2026, "coauthors": ["Y", "X"]},   # та же команда
        {"id": 3, "year": 2026, "coauthors": ["Z"]},        # phys 50% -> входит (не более 50%), команда {Z}
        {"id": 4, "year": 2026, "coauthors": []},           # без соавторов -> не считается
        {"id": 5, "year": 2026, "coauthors": ["W"]},        # phys 100% -> не междисциплинарная
    ]
    grader = FakeGrader({
        1: [DomainGrade("phys", 0.4, 1), DomainGrade("chem", 0.6, 1)],
        2: [DomainGrade("phys", 0.4, 1), DomainGrade("chem", 0.6, 1)],
        3: [DomainGrade("phys", 0.5, 1), DomainGrade("chem", 0.5, 1)],
        4: [DomainGrade("phys", 0.3, 1), DomainGrade("chem", 0.7, 1)],
        5: [DomainGrade("phys", 1.0, 1)],
    })
    e = make_profile(papers).calculate_expertise(grader, P.ExpertiseConfig(**CFG))["phys"]
    assert e.n_interdisciplinary_papers == 3 and e.n_teams == 2
    assert e.p_team == 3 * 2


def test_fwci_from_field_or_baselines():
    papers = [{"id": 1, "year": 2024, "coauthors": [], "fwci": 2.0},
              {"id": 2, "year": 2024, "coauthors": [], "citations": 10},
              {"id": 3, "year": 2025, "coauthors": [], "citations": 4}]      # нет базы для 2025 -> пропуск
    grader = FakeGrader({i: [DomainGrade("geo", 1.0, 0)] for i in (1, 2, 3)})
    baselines = P.build_citation_baselines([("geo", 2024, 2), ("geo", 2024, 8)])   # среднее 5
    e = make_profile(papers).calculate_expertise(grader, P.ExpertiseConfig(**CFG), baselines)["geo"]
    assert e.p_fwci == pytest.approx((2.0 + 10 / 5) / 2) and e.n_papers_with_fwci == 2


def test_total_score_formula():
    papers = [{"id": 1, "year": 2025, "coauthors": ["X"], "fwci": 1.5}]
    grader = FakeGrader({1: [DomainGrade("cs", 0.5, 4, 2), DomainGrade("math", 0.5, 3)]})
    cfg = P.ExpertiseConfig(k1=0.1, k2=0.2, k3=0.3, k4=0.4, decay=0.5, current_year=2026, top_n=None)
    e = make_profile(papers).calculate_expertise(grader, cfg)["cs"]
    p_ex = math.exp(-0.5 * 1 / 3)
    assert e.score == pytest.approx(6 * 0.1 + p_ex * 0.2 + 1.5 * 0.3 + (1 * 1) * 0.4)


@pytest.mark.parametrize("bad", [0, 1, -0.1, 1.5])
def test_coefficients_must_be_in_open_interval(bad):
    with pytest.raises(ValueError):
        P.ExpertiseConfig(k1=bad)


def test_publication_years_tracked():
    profile = make_profile([{"id": 1, "year": 2019, "coauthors": []}, {"id": 2, "year": 2024, "coauthors": []}])
    assert (profile.first_publication_year, profile.last_publication_year) == (2019, 2024)


# ----------------------------------------------------------------------------- top-n областей из domain_scores
def _scored_papers():
    return [
        {"id": 1, "year": 2026, "coauthors": ["X"], "scores": [{"domain": "phys", "score": 5}, {"domain": "chem", "score": 2}]},
        {"id": 2, "year": 2026, "coauthors": ["Y"], "scores": [{"domain": "phys", "score": 3}, {"domain": "bio", "score": 1}]},
    ]


def test_expertise_only_for_top_n_domains():
    grader = FakeGrader({
        1: [DomainGrade("phys", 0.3, 2), DomainGrade("chem", 0.7, 4)],
        2: [DomainGrade("phys", 0.4, 3), DomainGrade("bio", 0.6, 5)],
    })
    profile = make_profile(_scored_papers())           # domain_scores: phys 8, chem 2, bio 1
    expertise = profile.calculate_expertise(grader, P.ExpertiseConfig(**{**CFG, "top_n": 1}))
    assert list(expertise) == ["phys"]
    e = expertise["phys"]
    assert e.p_t == 2 + 3 and e.n_papers == 2
    # доли считаются по всем областям статьи: phys занимает 30% и 40% -> обе статьи междисциплинарные
    assert e.n_interdisciplinary_papers == 2 and e.n_teams == 2


def test_top_domain_without_graded_papers_gets_zero_record():
    grader = FakeGrader({1: [DomainGrade("phys", 1.0, 2)], 2: [DomainGrade("phys", 1.0, 3)]})
    expertise = make_profile(_scored_papers()).calculate_expertise(grader, P.ExpertiseConfig(**{**CFG, "top_n": 2}))
    assert set(expertise) == {"phys", "chem"}
    assert expertise["chem"].score == 0 and expertise["chem"].n_papers == 0


def test_top_n_requires_domain_scores():
    profile = make_profile([{"id": 1, "year": 2026, "coauthors": []}])   # статьи без "scores"
    with pytest.raises(ValueError):
        profile.calculate_expertise(FakeGrader({1: []}), P.ExpertiseConfig(k1=0.5, k2=0.5, k3=0.5, k4=0.5, top_n=3))


def test_top_n_validation():
    with pytest.raises(ValueError):
        P.ExpertiseConfig(top_n=0)
