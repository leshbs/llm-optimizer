# Severity Unknown Audit

Focused audit of `severity_unknown` (the single largest contradiction-feature coefficient, +1.092, H8 38.0) on the two required cases where it fired and G's prediction was wrong.

## Case id=619

**Guideline title:** Клинические рекомендации "Болезнь Крона"
Возрастная категория: Взрослые
# Лечение
## Консервативное лечение
### Легкая БК илеоцекальной локализации

**Patient protocol (first 400 chars):** Порядковый номер: 23
Код МКБ-10: K50.8
Пол: Женский
Возраст: 34
Дата приема: 2023-11-13
Жалобы: на дискомфорт в области п/о ран и стоме

Анамнез: на фоне лечения не большая положительная динамика 

Объективный статус: Общее состояние удовлетворительное. Кожные покровы и видимые слизистые обычной окраски.  Грудная клетка при пальпации б/б. ЧДД - 15 в мин.. Пульс удовлетворительного наполнения и нап

**Severity-related engineered features active:** ['severity_unknown']

**C2 prediction:** 1 (proba=0.899)

**G prediction:** 1 (proba=0.994)

**Ground truth:** 0

**Feature contribution:** `severity_unknown=1` (title specifies 'mild' Crohn's, narrative gives only minimal post-op discomfort with no explicit severity marker). `severity_unknown`'s learned coefficient (+1.092, the largest of any contradiction feature) pushes strongly toward Applicable whenever severity cannot be checked. Both C2 and G predict Applicable (1) here; **G's proba is far higher (0.994 vs. C2's own high-confidence FP)** -- the contradiction feature amplifies an error C2 already made, it does not introduce a new one.

**Audit conclusion:** the feature *behaves exactly as designed* -- 'severity unknown' correctly does not penalize applicability, following the project's own 'unknown != contradiction' principle. The row is a false positive for *both* models, and C2's own prediction here was already wrong before any H8 feature existed. **Classification: consistently harmful case, not a feature bug** -- the general policy behind `severity_unknown` is medically correct, but this specific narrow-evidence row is one where the correct general policy still produces the wrong local answer.

## Case id=722

**Guideline title:** Клинические рекомендации "Болезнь Крона"
Возрастная категория: Взрослые
# Лечение
## Консервативное лечение
### Легкая БК илеоцекальной локализации

**Patient protocol (first 400 chars):** Порядковый номер: 4
Код МКБ-10: K50.1
Пол: Мужской
Возраст: 49
Дата приема: 2022-03-10
Жалобы: Жалобы на сохраняющиеся достаточно выраженные боли в животе, по ходу толстого кишечника, больше слева, боли плохо снимаются НПВС, неустойчивый, неоформленный, жидковатый, учащённый стул, с примесью слизи и крови до 4-5 раз в сутки, слабость, головокружение.  В анамнезе стацлечение в колопроктологическом

**Severity-related engineered features active:** ['severity_contradiction']

**C2 prediction:** 1 (proba=0.818)

**G prediction:** 1 (proba=0.985)

**Ground truth:** 0

**Feature contribution:** `severity_contradiction=1` (title specifies 'mild' Crohn's; narrative describes pronounced pain and blood/mucus in stool 4-5x/day -- narrative severity genuinely contradicts the title). `severity_contradiction`'s learned coefficient is **positive (+0.491)**, a counter-intuitive sign: the feature correctly detects the contradiction, but the model has learned to treat that detection as evidence *for*, not against, applicability. Both C2 and G predict Applicable (1); ground truth is Not applicable (0).

**Audit conclusion:** this is the one clear **feature-direction bug** among the severity features -- `severity_contradiction` fires correctly (the lexical detection is right) but its learned coefficient sign is medically backwards for this row. Because it is a single linear coefficient fit across all 19 rows where `severity_contradiction` fires, the sign reflects the *net* effect across those rows, not a guarantee of correctness on every one. **Classification: feature bug (direction) on this case**, though see 53.0's audit note that the coefficient is fit correctly given the training data -- it is a limitation of a single global linear weight per feature, not a coding error in the feature's extraction logic.
