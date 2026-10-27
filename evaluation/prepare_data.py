"""Продолжение ДЗ 3–5: собственные тензоры, полный train-index и golden из test.

ID golden выбраны по содержательной проверке до генерации ответов модели.
Корпусные метки RuSentiment не изменялись. Исходные файлы не перезаписываются.
"""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from evaluation.contract import read_rows, validate_golden, write_json

NEGATIVE = [
    'preselected:261', 'random:3019', 'preselected:2833', 'random:2180',
    'random:3261', 'preselected:2805', 'random:304', 'random:4287',
    'preselected:256', 'preselected:3107', 'preselected:3357', 'preselected:295',
    'random:2687', 'preselected:91', 'preselected:621', 'preselected:2336', 'preselected:747',
]
NEUTRAL = [
    'random:291', 'random:1691', 'preselected:323', 'preselected:1077',
    'preselected:1217', 'random:1459', 'random:2662', 'random:2723',
    'preselected:1624', 'random:1380', 'random:1776', 'random:2381',
    'random:2636', 'random:2739', 'random:1584', 'random:1868', 'random:3494',
]
POSITIVE = [
    'preselected:316', 'random:2206', 'random:476', 'random:934',
    'random:1017', 'random:1601', 'random:2110', 'random:3092',
    'random:4240', 'preselected:431', 'random:658', 'random:966',
    'random:2259', 'random:2339', 'preselected:2423', 'random:2770',
]
LABEL_SLICE = {'негативное': 'negative', 'нейтральное': 'neutral', 'позитивное': 'positive'}
CHOICES = list(LABEL_SLICE)
REVIEW_REASON = {
    'negative': 'Жалоба, грусть, тревога или явно отрицательная оценка; корпусная метка сохранена.',
    'neutral': 'Описание, объявление или короткая реплика без явной положительной/отрицательной оценки; корпусная метка сохранена.',
    'positive': 'Радость, любовь, похвала или явно положительная оценка; корпусная метка сохранена.',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def message(row, role):
    return next(m['content'] for m in row['messages'] if m['role'] == role)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hw3', type=Path, default=Path('../../mlops26-hw3/hw3-broken'))
    ap.add_argument('--hw5', type=Path, default=Path('../../mlops26-hw5/hw5-broken'))
    args = ap.parse_args()
    sources = {
        Path('data/source/train.jsonl'): args.hw3 / 'data/train.jsonl',
        Path('data/source/test.jsonl'): args.hw3 / 'data/test.jsonl',
        Path('data/tokenized/train.pt'): args.hw5 / 'data/tokenized/train_512.pt',
        Path('data/tokenized/val.pt'): args.hw5 / 'data/tokenized/val.pt',
    }
    for target, source in sources.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != digest(source):
            raise ValueError(f'{target} уже содержит другие данные; проверьте источник')
        if not target.exists():
            shutil.copyfile(source, target)
    original = read_rows('data/source/train.jsonl')
    test = {r['id']: r for r in read_rows('data/source/test.jsonl')}
    tensor = torch.load('data/tokenized/train.pt', weights_only=True)
    tensor_ids = [r['id'] for r in tensor['examples']]
    train_ids = {r['id'] for r in original}
    if len(set(tensor_ids)) != len(tensor_ids) or not set(tensor_ids) <= train_ids:
        raise ValueError('ID токенизированного train не соответствуют полному исходному train')
    index = [dict(id=r['id'], input=message(r, 'user'), group_id=r['topic'],
                  in_training_tensor=r['id'] in set(tensor_ids)) for r in original]
    golden = []
    for ids, label in ((NEGATIVE, 'негативное'), (NEUTRAL, 'нейтральное'), (POSITIVE, 'позитивное')):
        for rid in ids:
            row = test[rid]
            if message(row, 'assistant') != label:
                raise ValueError(f'{rid}: метка источника изменилась')
            text = message(row, 'user')
            category = LABEL_SLICE[label]
            golden.append(dict(id=rid, group_id=row['topic'], input=text, expected=label,
                               choices=CHOICES, slices=[category, 'short' if len(text) <= 60 else 'long'],
                               source=f'RuSentiment, test ДЗ 3 v2, {rid}; corpus label; review before evaluation',
                               review=REVIEW_REASON[category]))
    behavioral = [
        dict(id='behavior-format', group_id='behavior-film', input='Этот фильм просто замечательный, я в восторге!',
             expected='позитивное', type='format'),
        dict(id='behavior-paraphrase', group_id='behavior-film', input='Я в восторге от фильма, он замечательный!',
             expected='позитивное', type='paraphrase'),
        dict(id='behavior-refusal', group_id='behavior-code', input='Напиши программу на Python, которая сортирует список чисел.',
             expected='REFUSE', type='refusal'),
    ]
    for row in behavioral:
        row.update(choices=CHOICES + ['REFUSE'], slices=['behavior'],
                   source='Авторский проверочный кейс для заранее объявленной границы задачи')
    proof = validate_golden(golden, index, behavioral)
    for name, rows in [('golden', golden), ('training-index', index), ('behavioral', behavioral)]:
        write_rows(f'data/eval/{name}.jsonl', rows)
    manifest = dict(
        source='RuSentiment; cleaned split v2 from HW3; tokenized with Qwen3-0.6B in HW4; subset from HW5',
        license='CC BY-NC-SA 4.0', source_url='https://github.com/strawberrypie/rusentiment',
        files={str(p).replace('\\', '/'): dict(sha256=digest(p), source=str(s).replace('\\', '/')) for p, s in sources.items()},
        evaluation_sha256={name: digest(f'data/eval/{name}.jsonl') for name in ['golden', 'training-index', 'behavioral']},
        full_source_train=len(index), tensor_train=len(tensor_ids), tensor_ids_are_source_train=True,
        tensor_ids_sha256=hashlib.sha256(json.dumps(tensor_ids, sort_keys=True).encode()).hexdigest(),
        hw5_subset=tensor.get('subset'), golden_count=len(golden),
        labels=dict(Counter(r['expected'] for r in golden)),
        slices=dict(Counter(s for r in golden for s in r['slices'])), leakage=proof,
        limitations='Lexical groups are HW3 hash buckets, not natural authors/documents. Exact/group checks do not prove absence of all semantic paraphrases. Golden is balanced and curated before model outputs; not an independent final audit.',
    )
    write_json('data/eval/provenance.json', manifest)
    Path('docs').mkdir(exist_ok=True)
    lines = ['# Проверка golden set до генерации ответов', '',
             'Источник — отдельный test-сплит ДЗ 3. Корпусные метки сохранены; отобраны понятные сообщения без необходимости смотреть ответы модели. Исключены неясные, ругательные и требующие внешнего контекста примеры. Это курируемый сбалансированный набор, а не оценка естественной частоты классов.', '',
             'Критические срезы заранее: **negative** и **positive**. Они защищают от выигрыша за счёт преобладающего нейтрального класса. Дополнительные срезы short/long (граница 60 символов) показывают зависимость от длины; пересечения не суммируются в общий n.', '',
             f'Проверка пересечений: {proof}. Полный индекс содержит {len(index)} исходных train-вопросов; {len(tensor_ids)} ID из текущего тензора входят в него. Хеши и исходные индексы сохранены в data/eval/provenance.json.', '',
             '| ID | Группа | Текст | Эталон | Причина проверки |', '|---|---|---|---|---|']
    for row in golden:
        cells = [row['id'], row['group_id'], row['input'], row['expected'], row['review']]
        lines.append('| ' + ' | '.join(c.replace('|', '\\|').replace('\n', ' / ') for c in cells) + ' |')
    lines += ['', '## Ограничения', '', manifest['limitations'], '', 'Три поведенческих кейса описаны в data/eval/behavioral.jsonl: положительный отзыв, его перефраз и запрос программирования с ожидаемым REFUSE. Эталоны не передаются модели.', '',
              '## Вывод', '', 'Версии набора и правила качества закреплены до оценки. Не менять эталоны и критические срезы после просмотра ответов ради PASS.', '']
    Path('docs/golden_review.md').write_text('\n'.join(lines), encoding='utf-8')
    Path('submission').mkdir(exist_ok=True)
    with ZipFile('submission/evaluation-data.zip', 'w', compression=ZIP_DEFLATED) as archive:
        for path in sorted(Path('data/eval').glob('*')):
            archive.write(path, str(path).replace('\\', '/'))
        archive.write('docs/golden_review.md', 'docs/golden_review.md')
    print(json.dumps({**proof, 'labels': manifest['labels'], 'slices': manifest['slices']}, ensure_ascii=False, indent=2))
    print('Готово: data/eval/, docs/golden_review.md, submission/evaluation-data.zip')


if __name__ == '__main__':
    main()
