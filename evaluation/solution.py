"""Строгий подсчёт ответов и запрет выпуска при несовместимости или регрессии."""

import json
import math
from collections import defaultdict


def _indexed(items, label):
    if not isinstance(items, list) or not items:
        raise ValueError(f'{label}: нужен непустой список')
    result = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id']:
            raise ValueError(f'{label}: нужен строковый id')
        if item['id'] in result:
            raise ValueError(f'{label}: повторный id {item["id"]}')
        result[item['id']] = item
    return result


def _names(names):
    if not isinstance(names, list) or not names or any(not isinstance(s, str) or not s for s in names):
        raise ValueError('Нужен непустой список имён срезов')
    if len(names) != len(set(names)):
        raise ValueError('Повторное имя среза')
    return names


def _stats(items):
    n = len(items)
    correct = sum(item['correct'] for item in items)
    valid = sum(item['valid'] for item in items)
    errors = sum(item['error'] is not None for item in items)
    return dict(n=n, correct=correct, valid=valid, errors=errors,
                accuracy=correct / n, format_rate=valid / n)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Повторный ключ JSON')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f'Недопустимая константа JSON: {value}')


def score(rows, outputs):
    """Верните overall, slices и per_item. Не удаляйте плохие ответы из знаменателя.

    Строгий ответ: JSON с единственным строковым полем answer из choices.
    outputs: [{id, raw, error, latency_seconds}, ...].
    В overall и каждом срезе: n, correct, valid, errors, accuracy, format_rate.
    В per_item: id, raw, parsed, valid, correct, error, slices.
    Несовпадающие/дублирующиеся id и пустой набор должны вызвать ValueError.
    """
    expected_rows = _indexed(rows, 'rows')
    actual = _indexed(outputs, 'outputs')
    if set(expected_rows) != set(actual):
        raise ValueError('ID заданий и ответов не совпадают')
    per_item, slices = [], defaultdict(list)
    for row in rows:
        names = _names(row.get('slices'))
        choices = row.get('choices')
        if not isinstance(choices, list) or not choices or any(not isinstance(c, str) for c in choices):
            raise ValueError('Нужны допустимые строковые choices')
        if row.get('expected') not in choices:
            raise ValueError('Эталон отсутствует в choices')
        output = actual[row['id']]
        raw, error = output.get('raw', ''), output.get('error')
        if error is not None and (not isinstance(error, str) or not error):
            raise ValueError('error должен быть непустой строкой или None')
        parsed, valid = None, False
        if error is None and isinstance(raw, str):
            try:
                value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
                if isinstance(value, dict) and set(value) == {'answer'} and isinstance(value['answer'], str) and value['answer'] in choices:
                    parsed, valid = value['answer'], True
            except (ValueError, TypeError):
                pass  # Плохой ответ остаётся в знаменателе, это не ошибка контракта.
        item = dict(id=row['id'], raw=raw, parsed=parsed, valid=valid,
                    correct=valid and parsed == row['expected'], error=error, slices=list(names))
        per_item.append(item)
        for name in names:
            slices[name].append(item)
    return dict(overall=_stats(per_item),
                slices={name: _stats(items) for name, items in sorted(slices.items())},
                per_item=per_item)


def _validate_counts(actual, expected, label):
    if not isinstance(actual, dict):
        raise ValueError(f'{label}: отсутствуют метрики')
    for key in ('n', 'correct', 'valid', 'errors'):
        value = actual.get(key)
        if type(value) is not int or value != expected[key]:
            raise ValueError(f'{label}: несогласованный счётчик {key}')
    for key in ('accuracy', 'format_rate'):
        value = actual.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{label}: неконечная метрика {key}')
        if not math.isclose(value, expected[key], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f'{label}: метрика {key} не соответствует counts')


def _validate_score(report, label):
    if not isinstance(report, dict):
        raise ValueError(f'{label}: отчёт отсутствует')
    items = report.get('per_item')
    indexed = _indexed(items, label)
    groups = defaultdict(list)
    for item in items:
        if not {'id', 'raw', 'parsed', 'valid', 'correct', 'error', 'slices'} <= set(item):
            raise ValueError(f'{label}: неполная строка per_item')
        if type(item.get('valid')) is not bool or type(item.get('correct')) is not bool:
            raise ValueError(f'{label}: нужны булевы valid/correct')
        if 'error' not in item or (item['error'] is not None and (not isinstance(item['error'], str) or not item['error'])):
            raise ValueError(f'{label}: некорректное поле error')
        if item['correct'] and not item['valid']:
            raise ValueError(f'{label}: невалидный ответ отмечен верным')
        if item['error'] is not None and (item['valid'] or item['correct']):
            raise ValueError(f'{label}: ошибка генерации отмечена успешным ответом')
        for name in _names(item.get('slices')):
            groups[name].append(item)
    _validate_counts(report.get('overall'), _stats(items), label)
    if not isinstance(report.get('slices'), dict) or set(report['slices']) != set(groups):
        raise ValueError(f'{label}: имена срезов не соответствуют per_item')
    for name, members in groups.items():
        _validate_counts(report['slices'][name], _stats(members), f'{label}/{name}')
    return indexed


def quality_gate(baseline, candidate, critical_slices):
    """Верните {passed: True, ...} или вызовите ValueError с причиной отказа.

    Проверьте совместимость протокола/набора/baseline, метрики и counts,
    все id и срезы, общую accuracy/format_rate, критические срезы и behavior.
    Missing, NaN, пустой набор и ошибки генерации означают отказ.
    Пороги задаются до измерения; учебный допуск регрессии равен нулю.
    """
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise ValueError('Нужны два отчёта')
    for key in ('protocol_sha256', 'golden_sha256', 'baseline_spec_sha256'):
        if not isinstance(baseline.get(key), str) or not baseline[key] or candidate.get(key) != baseline[key]:
            raise ValueError(f'Несовместимый или отсутствующий {key}')
    indexed = []
    behavior = []
    for label, report in (('baseline', baseline), ('candidate', candidate)):
        if report.get('status') != 'complete' or not isinstance(report.get('model'), dict) or not report['model'].get('kind'):
            raise ValueError(f'{label}: отчёт не завершён или отсутствует модель')
        indexed.append(_validate_score(report, label))
        if report['overall']['n'] < 50 or len(report['slices']) < 3:
            raise ValueError(f'{label}: нужно минимум 50 golden-примеров и три среза')
        behavior.append(_validate_score(report.get('behavioral'), f'{label}/behavioral'))
        if report['behavioral']['overall']['n'] < 3:
            raise ValueError('Нужно минимум три behavioral-кейса')
        if report['overall']['errors'] or report['behavioral']['overall']['errors']:
            raise ValueError(f'{label}: есть ошибки генерации')
    for left, right in ((indexed[0], indexed[1]), (behavior[0], behavior[1])):
        if set(left) != set(right):
            raise ValueError('Различаются id baseline/candidate')
        if any(set(left[i]['slices']) != set(right[i]['slices']) for i in left):
            raise ValueError('Различаются срезы одних и тех же заданий')
    if set(baseline['slices']) != set(candidate['slices']):
        raise ValueError('Различаются имена срезов')
    for name in baseline['slices']:
        if baseline['slices'][name]['n'] != candidate['slices'][name]['n']:
            raise ValueError(f'Различаются размеры среза {name}')
    for key in ('accuracy', 'format_rate'):
        if candidate['overall'][key] < baseline['overall'][key]:
            raise ValueError(f'Регрессия общей метрики {key}')
    for name in _names(critical_slices):
        if name not in baseline['slices'] or name not in candidate['slices']:
            raise ValueError(f'Отсутствует критический срез {name}')
        if candidate['slices'][name]['accuracy'] < baseline['slices'][name]['accuracy']:
            raise ValueError(f'Регрессия в критическом срезе {name}')
    if candidate['behavioral']['overall']['correct'] != candidate['behavioral']['overall']['n']:
        raise ValueError('Не все обязательные behavioral-кейсы пройдены')
    return dict(passed=True, baseline_accuracy=baseline['overall']['accuracy'],
                candidate_accuracy=candidate['overall']['accuracy'],
                critical_slices=list(critical_slices), behavioral_passed=True)
