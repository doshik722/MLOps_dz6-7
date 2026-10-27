"""Один отчёт A+B из существующих результатов. Отсутствующие замеры не выдумываются."""
import csv
import json
import os
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from evaluation.contract import read_json


def optional(path):
    path = Path(path)
    return read_json(path) if path.exists() else None


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' / ').replace('<', '&lt;').replace('>', '&gt;')


def main():
    Path('docs').mkdir(exist_ok=True)
    source = optional('data/eval/provenance.json') or {}
    sweep = optional('metrics/sweep.json')
    candidate = optional('metrics/candidate-model.json')
    baseline = optional('evaluation/baseline.json')
    a, b = optional('metrics/baseline-eval.json'), optional('metrics/candidate-eval.json')
    gate, negative = optional('metrics/quality-gate.json'), optional('metrics/negative-gate.json')
    storage = optional('metrics/storage-check.json')
    released = optional('metrics/released.json')
    imported = optional('metrics/offline-import.json')
    verified = optional('metrics/verification.json')
    config = read_json('evaluation/params.json')
    table = []
    if Path('docs/comparison.csv').exists():
        with Path('docs/comparison.csv').open(encoding='utf-8', newline='') as stream:
            table = list(csv.DictReader(stream))
    lines = ['# ДЗ 6+7 — эксперименты и качество классификации тональности', '',
             '## Условия и протокол, закреплённые до оценки', '',
             'Продолжение ДЗ 3–5: RuSentiment, Qwen3-0.6B, CPU/float32, seed 42. Бюджет каждого из шести вариантов одинаков: 4 шага, train-пул 32, val 8, batch 1 × accumulation 2. Ранг 4/8, lr 1e-4/3e-4, q/v или все семь проекций. alpha/r=2. Короткий поиск проверяет пайплайн; он не доказывает оптимальность гиперпараметров.', '',
             f"Источник: полный raw train-index {source.get('full_source_train', 'не подготовлен')}, текущий train-тензор {source.get('tensor_train', 'не подготовлен')} строк. Golden — 50 примеров из test ДЗ 3; эталоны и источник записаны в docs/golden_review.md. Контроль пересечений: {source.get('leakage', 'ещё не выполнялся')}.", '',
             'До генерации назначены критические срезы negative и positive: средняя accuracy не должна скрывать ухудшение эмоциональных классов за счёт нейтрального. Дополнительные short/long разделены границей 60 символов. Срезы пересекаются, их n не складывается. Это повторно используемый golden, а не независимый финальный аудит.', '',
             'Одинаковая system-инструкция и greedy-генерация для baseline/candidate. Формат — JSON с единственным полем answer; допустимые русские метки или REFUSE для неподдерживаемой задачи. Expected не входит в prompt. Бюджет 32 новых токена. Никакой допуск регрессии после измерения не добавляется.', '',
             '## Часть A: шесть экспериментов и выбор по loss', '']
    if table:
        lines += ['Таблица получена из MLflow командой make report, а не заполнена вручную.', '',
                  '| Имя | Run ID | r | lr | Val базы | Итоговый val | Время, с | Память, МБ | Папка адаптера, МБ |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in table:
            values = [row[k] for k in ['name', 'run_id', 'r', 'lr']]
            values += [f"{float(row[k]):.6f}" for k in ['base_val_loss', 'final_val_loss']]
            values += [f"{float(row[k]):.1f}" for k in ['seconds', 'peak_memory_mb', 'adapter_mb']]
            lines.append('| ' + ' | '.join(cell(v) for v in values) + ' |')
        lines += ['', f"Минимальный val loss у `{table[0]['name']}`: {table[0]['final_val_loss']}. Порог допуска в candidate: final/base ≤ 1; test/golden в выборе не использовались.", '']
        lines += ['Память — максимум выбранных отсчётов RSS всего Python-процесса, а не непрерывный пик отдельной модели. Последовательные runs наследуют кэш аллокатора и импортированных библиотек; по этим значениям нельзя ранжировать чистую память адаптеров. Время относится к обучению/валидации, без загрузки базы и упаковки MLflow. adapter_mb включает токенизатор и manifest, поэтому это размер всей папки адаптера.', '']
    else:
        lines += ['Шесть запусков ещё не выполнены. comparison.csv и измеренные числа появятся после make sweep и make report.', '']
    lines += [f'Кандидат: `{json.dumps(candidate, ensure_ascii=False)}`' if candidate else 'Версия candidate пока не зарегистрирована.', '',
              f'Закреплённый baseline: `{json.dumps(baseline, ensure_ascii=False)}`' if baseline else 'Baseline пока не инициализирован.', '',
              '## Часть B: реальные ответы и метрики', '']
    if a and b:
        lines += ['| Версия | n | Верных | Валидных | Ошибок генерации | Accuracy | Format rate |', '|---|---:|---:|---:|---:|---:|---:|']
        for label, report in [('baseline', a), ('candidate', b)]:
            m = report['overall']
            lines.append(f"| {label} | {m['n']} | {m['correct']} | {m['valid']} | {m['errors']} | {m['accuracy']:.2%} | {m['format_rate']:.2%} |")
        lines += ['', '| Срез | n baseline/candidate | Верных baseline/candidate | Accuracy baseline/candidate |', '|---|---|---|---|']
        for name in sorted(set(a['slices']) | set(b['slices'])):
            left, right = a['slices'][name], b['slices'][name]
            lines.append(f"| {name} | {left['n']}/{right['n']} | {left['correct']}/{right['correct']} | {left['accuracy']:.2%}/{right['accuracy']:.2%} |")
        diff = optional('metrics/eval-diff.json') or {}
        lines += ['', f"Исправленные ID: {diff.get('improved', [])}.", f"Новые ошибки: {diff.get('regressed', [])}.", '', '### Разбор пяти пар ответов', '']
        examples = {r['id']: r for r in (read_jsonl(config['golden']))}
        before, after = {r['id']: r for r in a['per_item']}, {r['id']: r for r in b['per_item']}
        selected = list(dict.fromkeys(diff.get('regressed', [])[:2] + diff.get('improved', [])[:2] + list(examples)))[:5]
        for rid in selected:
            row, left, right = examples[rid], before[rid], after[rid]
            lines += [f'#### {rid}', '', row['input'], '',
                      f"Эталон: `{row['expected']}`. Baseline: `{cell(left['raw'])}`; candidate: `{cell(right['raw'])}`.",
                      f"Верность: {left['correct']} → {right['correct']}; валидность: {left['valid']} → {right['valid']}. Строгий формат и правильность метки оцениваются отдельно.", '']
        lines += ['### Поведенческие кейсы', '', '| ID | Тип | Эталон | Ответ кандидата | Прошёл |', '|---|---|---|---|---|']
        behavior = {r['id']: r for r in b['behavioral']['per_item']}
        for row in read_jsonl(config['behavioral']):
            item = behavior[row['id']]
            lines.append(f"| {row['id']} | {row['type']} | {row['expected']} | {cell(item['raw'])} | {item['correct']} |")
    else:
        lines += ['Реальная генерация ещё не выполнялась. Результаты синтетических unit-тестов не являются метриками Qwen. После make evaluate здесь появятся общие метрики, срезы, ошибки и разбор пар.', '']
    lines += ['', '## Допуск, отрицательная проверка и хранение', '',
              f"Quality gate: `{json.dumps(gate, ensure_ascii=False)}`" if gate else 'Quality gate ещё не запускался.', '',
              f"Отрицательная проверка: `{json.dumps(negative, ensure_ascii=False)}`" if negative else 'Отрицательная проверка ещё не выполнялась.', '',
              f"Хранение: `{json.dumps(storage, ensure_ascii=False)}`" if storage else 'Сохранность артефакта после force-recreate контейнера ещё не подтверждена.', '']
    if candidate:
        try:
            os.environ.setdefault('MLFLOW_HTTP_REQUEST_TIMEOUT', '5')
            os.environ.setdefault('MLFLOW_HTTP_REQUEST_MAX_RETRIES', '1')
            from evaluation.cli import tracking
            p, client = tracking()
            version = client.get_model_version_by_alias(candidate['name'], 'champion')
            lines += [f'Текущий champion: models:/{version.name}/{version.version}; run={version.run_id}.', '']
        except Exception as exc:
            lines += [f'Чтение champion: {type(exc).__name__}: {exc}. При непройденном gate новый champion не назначается.', '']
    lines += ['## Ссылки на MLflow и воспроизводимость', '']
    if imported:
        lines += ['Из-за первоначального сбоя Docker шесть обучений сначала записаны в локальный MLflow/SQLite. Затем реальные runs перенесены на HTTP-сервер без повторного обучения: сохранены параметры, вся история метрик с исходными timestamps, source snapshot и файлы моделей. Каждый файл кандидатов сверен по SHA256. Соответствие исходных и серверных IDs и хеши — metrics/offline-import.json. Внутренний run_id в неизменённом MLmodel хранит происхождение пакета; реестр ссылается на серверный run.', '']
    if verified:
        lines += [f"Приёмка исходными тестами: {verified.get('acceptance')}; unit-тесты: {verified.get('unit_tests')}; отпечаток {verified.get('tests_fingerprint')}.", '']
    uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5006')
    if uri.startswith(('http://', 'https://')) and sweep:
        from mlflow import MlflowClient
        client = MlflowClient(tracking_uri=uri)
        for run_id in [sweep['parent_id'], *[r['run_id'] for r in sweep['rows']], *([b['evaluation_run_id']] if b else [])]:
            run = client.get_run(run_id)
            lines.append(f'- [Run {run_id}]({uri}/#/experiments/{run.info.experiment_id}/runs/{run_id})')
    else:
        lines += [f'Текущее tracking-хранилище: `{uri}`. Это локальная резервная запись; доступность Docker-сервера и сохранность volume проверяются отдельно.']
    lines += ['', 'SHA256 кода, конфига и файлов данных записаны в metrics/sweep.json и tags runs; исходный Git-коммит записан в provenance. Протокол оценки, golden и baseline имеют отдельные SHA256. Исходные tests/ не изменены: c42ac7788026.', '', '## Ограничения оценки и модельный судья', '',
              'Accuracy проверяет совпадение метки с проверенным эталоном, format rate — пригодность ответа к автоматическому разбору. BLEU/ROUGE/BERTScore могут считать похожими ответы с противоположными метками и не заменяют эту проверку. На 50 курируемых примерах нельзя уверенно переносить небольшую разницу на весь поток данных; важны n и новые ошибки.', '',
              'LLM-судья мог бы помочь разобрать сарказм и неоднозначные социальные сообщения по явной рубрике. Возможны смещения к порядку A/B, стилю и многословию, ошибки на иронии и контексте. Нужны скрытые имена моделей, перестановка A/B, сохранённые вердикты и проверка спорных случаев человеком. Платный API не использовался, judge не заменяет scorer меток.', '',
              '## Вывод', '']
    if (released and gate and gate.get('passed') and b
            and released.get('model') == b['model']
            and released.get('evaluation_run_id') == b.get('evaluation_run_id')
            and released.get('protocol_sha256') == b.get('protocol_sha256')):
        lines += [f"Кандидат принят после PASS и назначен champion: {released['model']['uri']}; eval run {released['evaluation_run_id']}."]
    elif gate:
        lines += ['Кандидат прошёл gate, но продвижение ещё не выполнено.' if gate.get('passed') else f"Кандидат не принят: {gate.get('reason')}. Baseline не подменяется, champion не переназначается. Следующий шаг — разбор сохранённых ошибок; условия текущего сравнения не ослабляются."]
    else:
        lines += ['Работа подготовлена, но решение о выпуске ещё не получено. Для завершения нужны реальные шесть runs, генерация двух версий, проверка качества и сохранности. Отсутствующие результаты нельзя считать успешной сдачей.']
    Path('docs/conclusion.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    defects = ['# ДЗ 6+7 — разбор четырёх дефектов', '',
               '## 1. Общий run смешивал разные конфигурации', '',
               'В src/sweep.py все шесть вариантов выполнялись внутри all-candidates. Параметры логировались только при пустом run, поэтому фактическая модель и подпись эксперимента расходились. Теперь каждый кандидат получает собственный nested run внутри родительского sweep; параметры логируются без скрывающего ошибки условия.', '',
               f"Доказательство: {'получено ' + str(len(sweep['rows'])) + ' отдельных строк в metrics/sweep.json' if sweep else 'реальный sweep ещё не выполнен'}. Проверки 1, 3–5 сверяют отдельные runs, бюджет, реальные файлы адаптеров и одинаковые промпты.", '',
               '## 2. Артефакты пропадали после пересоздания контейнера', '',
               'В compose.yaml volume artifacts объявлен, но не подключён к mlflow. Postgres сохранял метаданные, а файлы моделей оставались в слое контейнера. Добавлен mount artifacts:/mlartifacts, соответствующий --artifacts-destination. Перезапуск процесса не доказывает сохранность: нужен force-recreate и проверка SHA256 probe-файла.', '',
               f"Доказательство хранения: {json.dumps(storage, ensure_ascii=False) if storage else 'ещё не измерено; выполнить make check-storage после запуска Docker'}. Volumes и база при проверке не удаляются.", '',
               '## 3. Версии данных исключались из provenance', '',
               'provenance() уже вычисляла data_train_sha256 и data_val_sha256, но sweep отбрасывал ключи data_. Теперь полная provenance передаётся в parent и дочерние runs. Логируются Git SHA, dirty-статус, snapshot src, uv.lock, SHA256 кода/конфига/train/val и точная ревизия Hugging Face. Изменение файла по тому же пути больше не считается тем же экспериментом.', '',
               f"Происхождение собственных данных: data/eval/provenance.json; полный train-index {source.get('full_source_train')}, тензор {source.get('tensor_train')}, golden 50. Версии файлов — data.dvc; доступ к проверочным данным — submission/evaluation-data.zip. Точное/групповое пересечение: {source.get('leakage')}. Семантическое отсутствие всех перефразов этим не доказано.", '',
               '## 4. Порог не проверялся до регистрации', '',
               'src/promote.py вычислял ratio, но не вызывал gate(): даже заведомо плохой run мог создать версию и переназначить alias. Теперь gate(run, threshold) вызывается до register_model. Проверяются FINISHED, конечность метрик, положительный base loss и final/base ≤ заранее заданного порога. Нулевой порог обязан отклонить положительный loss без побочных записей в реестр.', '',
               'Часть B дополнительно требует строгий quality_gate до смены champion: одинаковые версии протокола и данных, полные непротиворечивые counts, отсутствие ошибок генерации, отсутствие общей/критической регрессии и прохождение трёх behavioral-кейсов. Отказ допустим и не обходится переименованием кандидата или изменением baseline.', '',
               f"Доказательство отрицательного теста: {json.dumps(negative, ensure_ascii=False) if negative else 'ожидает make check-negative на реальных ответах'}. Unit-тесты выполняются отдельно и не выдаются за реальную оценку модели.", '',
               '## Переносимость Windows', '',
               'В чтении/записи конфигурации, manifest и оценки явно задан UTF-8; изолированный predict выводит ASCII JSON. Makefile включает UTF-8 и немедленный вывод; scripts/run_command.py находит Docker Desktop и Git Bash без ручной настройки PATH. scripts/normalize_packages.py исправляет Windows-пути внутри MLmodel: artifacts/adapter и упакованный model.py вместо обратных слешей/абсолютного пути; веса не меняются. Это обнаруженная реальной проверкой проблема переносимости MLflow 2.22. CPU/float32 выбран согласно методичке; бюджет поиска не увеличен. Тесты не изменены, выданный общий отпечаток c42ac7788026.', '',
               'В отрицательной проверке evaluation/cli.py портится копия raw-ответов, затем score пересчитывает все counts. Ручная подмена одного счётчика удалена, чтобы отказ доказывал ухудшение ответов, а не внутреннюю несогласованность отчёта. При первоначальном сбое Docker добавлен scripts/import_offline.py: перенос реальных runs с полной историей метрик и побайтовой проверкой артефактов. Он не обучает модель и не меняет измеренные значения.', '',
               '## Вывод', '',
               'Исправлены четыре причины ошибок учёта/хранения/версий/допуска и реализованы score/quality_gate. Фактические результаты запуска, включая отсутствующие этапы и отказы, отражены в docs/conclusion.md.', '']
    Path('docs/defects.md').write_text('\n'.join(defects), encoding='utf-8')
    submission = Path('submission/results')
    submission.mkdir(parents=True, exist_ok=True)
    for path in [*Path('metrics').glob('*.json'), *Path('docs').glob('*.md'), *Path('docs').glob('*.csv')]:
        target = submission / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    with ZipFile('submission/results.zip', 'w', compression=ZIP_DEFLATED) as archive:
        for path in sorted(submission.rglob('*')):
            if path.is_file():
                archive.write(path, str(path.relative_to(submission)).replace('\\', '/'))
    print('Обновлены docs/defects.md, docs/conclusion.md и submission/results.zip; отсутствующие этапы обозначены явно.')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


if __name__ == '__main__':
    main()
