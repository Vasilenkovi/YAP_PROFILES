import math
from dataclasses import dataclass, field
from datetime import date

# Контракт модуля taxonomyGrader (используется в Profile.calculate_expertise):
#   grader.grade(paper) -> список оценок статьи по предметным областям; у каждой оценки атрибуты:
#     domain         — предметная область (кластер таксономии);
#     term_share     — доля терминов статьи, отнесённых к кластеру этой области (0..1);
#     term_depth_sum — сумма глубин терминов (ключевых слов), встроенных в таксономию области;
#     tool_depth_sum — то же для именованных сущностей-инструментов (таксономия инструментов).
#   Можно вернуть все области: отбор областей, покрывающих >= 95% терминов, выполняет Profile.


# ----------------------------------------------------------------------------- параметры и результат
@dataclass
class ExpertiseConfig:
    """Параметры формулы E_f = P_t*k1 + P_ex*k2 + P_FWCI*k3 + P_team*k4."""
    k1: float = 0.25                      # вес P_t (таксономии)
    k2: float = 0.25                      # вес P_ex (опыт и активность)
    k3: float = 0.25                      # вес P_FWCI (признание)
    k4: float = 0.25                      # вес P_team (междисциплинарность)
    decay: float = 1.0                    # l — коэффициент затухания в P_ex
    current_year: int = field(default_factory=lambda: date.today().year)
    coverage_threshold: float = 0.95      # области статьи: суммарно не менее 95% терминов
    interdisciplinary_max_share: float = 0.5   # P_team: в области не более 50% терминов статьи
    tool_weight: float = 1.0              # вклад таксономии инструментов в P_t
    top_n: int | None = 5                 # экспертиза только для top-n областей из domain_scores; None — все

    def __post_init__(self):
        for name in ("k1", "k2", "k3", "k4"):
            value = getattr(self, name)
            if not 0 < value < 1:
                raise ValueError(f"{name} должен быть в интервале (0, 1), получено {value}")
        if self.decay < 0:
            raise ValueError("decay (l) не может быть отрицательным")
        if not 0 < self.coverage_threshold <= 1:
            raise ValueError("coverage_threshold должен быть в (0, 1]")
        if self.top_n is not None and self.top_n < 1:
            raise ValueError("top_n должен быть >= 1 или None")


@dataclass
class DomainExpertise:
    """Экспертиза автора в одной области: итоговая оценка и её составляющие."""
    domain: str
    score: float
    p_t: float
    p_ex: float
    p_fwci: float
    p_team: float
    n_papers: int
    n_papers_with_fwci: int
    n_interdisciplinary_papers: int
    n_teams: int


def build_citation_baselines(records):
    """Средняя цитируемость по (область, год) для расчёта FWCI.

    records — итерируемое из (domain, year, citations) по всему корпусу статей
    (не только по одному автору), иначе FWCI теряет смысл.
    """
    sums, counts = {}, {}
    for domain, year, citations in records:
        if year is None or citations is None:
            continue
        key = (domain, int(year))
        sums[key] = sums.get(key, 0.0) + float(citations)
        counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums}


class Profile:
    def __init__(self, name, age, email):
        self.name = name
        self.age = age
        self.email = email
        self.orcid = None
        self.papers = []
        self.coauthors = {}
        self.domain_scores = {}
        self.affiliation = None
        self.expertise = {}
        self.first_publication_year = None
        self.last_publication_year = None

    def upload_paper(self, paper):
        for domain_score in paper.get("scores", []):
            if domain_score["domain"] in self.domain_scores:
                self.domain_scores[domain_score["domain"]] += domain_score["score"]
            else:
                self.domain_scores[domain_score["domain"]] = domain_score["score"]
        self.papers.append(paper)
        for coauthor in paper.get("coauthors", []):
            if coauthor not in self.coauthors:
                self.coauthors[coauthor] = 1
            else:
                self.coauthors[coauthor] += 1
        year = paper.get("year")
        if year is not None:
            if self.first_publication_year is None or year < self.first_publication_year:
                self.first_publication_year = year
            if self.last_publication_year is None or year > self.last_publication_year:
                self.last_publication_year = year

    def get_top_coauthors(self, n=5):
        sorted_coauthors = sorted(self.coauthors.items(), key=lambda x: x[1], reverse=True)
        return sorted_coauthors[:n]

    def get_top_domains(self, n=5):
        sorted_domains = sorted(self.domain_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_domains[:n]

    def get_top_expertise(self, n=5):
        sorted_expertise = sorted(self.expertise.values(), key=lambda e: e.score, reverse=True)
        return sorted_expertise[:n]

    # ------------------------------------------------------------------------- экспертиза
    def calculate_expertise(self, grader, config=None, citation_baselines=None):
        """Экспертиза по top-n областям автора из domain_scores: E_f = P_t*k1 + P_ex*k2 + P_FWCI*k3 + P_team*k4.

        Области берутся из get_top_domains(config.top_n) (config.top_n=None — все области, найденные
        taxonomyGrader). Идентификаторы областей в paper["scores"] и в ответе grader должны совпадать.

        grader             — объект taxonomyGrader с методом grade(paper) (контракт — в начале модуля);
        config             — ExpertiseConfig (коэффициенты k1..k4, l, текущий год, пороги);
        citation_baselines — {(domain, year): средняя цитируемость} для FWCI, см. build_citation_baselines;
                             если у статьи уже есть поле "fwci", используется оно.

        Поля статьи: "year", "coauthors" (без самого автора), "citations" и/или "fwci".
        Результат сохраняется в self.expertise и возвращается: {domain: DomainExpertise}.
        """
        config = config or ExpertiseConfig()
        top_domains = None
        if config.top_n is not None:
            top_domains = [domain for domain, _ in self.get_top_domains(config.top_n)]
            if not top_domains:
                raise ValueError("domain_scores пуст: загрузите статьи с полем 'scores' или задайте top_n=None")

        def new_acc():
            return {"p_t": 0.0, "p_ex": 0.0, "fwci": [], "n_papers": 0,
                    "interdisciplinary_papers": 0, "teams": set()}

        # top-области без единой статьи в ответе grader тоже попадут в результат (с нулевой оценкой)
        per_domain = {domain: new_acc() for domain in top_domains} if top_domains else {}

        for paper in self.papers:
            # области статьи и доли терминов — по всем областям, фильтр top-n применяется после
            selected = self._select_domains(grader.grade(paper), config.coverage_threshold)
            team = frozenset(c for c in paper.get("coauthors", []) if c != self.orcid)
            for grade in selected:
                if top_domains is not None and grade.domain not in per_domain:
                    continue
                acc = per_domain.setdefault(grade.domain, new_acc())
                acc["n_papers"] += 1
                # P_t: глубины терминов, встроенных в таксономию области
                acc["p_t"] += grade.term_depth_sum + config.tool_weight * grade.tool_depth_sum
                # P_ex (формула 2): вклад статьи затухает с её возрастом
                acc["p_ex"] += self._recency_weight(paper.get("year"), config)
                fwci = self._paper_fwci(paper, grade.domain, citation_baselines)
                if fwci is not None:
                    acc["fwci"].append(fwci)
                # P_team: статья в соавторстве, где на эту область приходится не более 50% терминов
                if team and grade.term_share <= config.interdisciplinary_max_share:
                    acc["interdisciplinary_papers"] += 1
                    acc["teams"].add(team)

        self.expertise = {}
        for domain, acc in per_domain.items():
            p_fwci = sum(acc["fwci"]) / len(acc["fwci"]) if acc["fwci"] else 0.0
            p_team = acc["interdisciplinary_papers"] * len(acc["teams"])
            score = (acc["p_t"] * config.k1 + acc["p_ex"] * config.k2
                     + p_fwci * config.k3 + p_team * config.k4)
            self.expertise[domain] = DomainExpertise(
                domain=domain, score=score, p_t=acc["p_t"], p_ex=acc["p_ex"], p_fwci=p_fwci, p_team=p_team,
                n_papers=acc["n_papers"], n_papers_with_fwci=len(acc["fwci"]),
                n_interdisciplinary_papers=acc["interdisciplinary_papers"], n_teams=len(acc["teams"]),
            )
        return self.expertise

    @staticmethod
    def _select_domains(grades, coverage_threshold):
        """Области статьи: по убыванию доли терминов, пока суммарно не наберётся порог (95%).

        Если таксономии в сумме покрывают меньше порога, берутся все области с ненулевой долей.
        """
        selected, covered = [], 0.0
        for grade in sorted(grades, key=lambda g: g.term_share, reverse=True):
            if grade.term_share <= 0 or covered >= coverage_threshold:
                break
            selected.append(grade)
            covered += grade.term_share
        return selected

    @staticmethod
    def _recency_weight(year, config):
        if year is None:
            return 0.0
        age = max(0, config.current_year - int(year))   # статьи "из будущего" не усиливаются
        return math.exp(-config.decay * age / 3)

    @staticmethod
    def _paper_fwci(paper, domain, citation_baselines):
        if paper.get("fwci") is not None:
            return float(paper["fwci"])
        if citation_baselines is None or paper.get("citations") is None or paper.get("year") is None:
            return None
        expected = citation_baselines.get((domain, int(paper["year"])))
        if not expected:
            return None
        return float(paper["citations"]) / expected
