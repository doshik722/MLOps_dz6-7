"""Восстановление шести реальных runs из локального MLflow на Docker-сервер.

Никакого обучения/подмены метрик: сохраняются исходные timestamps и все
артефакты. Новые run IDs связаны с исходными tags.offline_run_id и JSON-картой.
До публикации результатов каждый файл кандидатов сверяется по SHA256.
"""
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunTag


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def hashes(root):
    result = {}
    for path in sorted(Path(root).rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                result[path.relative_to(root).as_posix()] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result


def main():
    import yaml
    params = yaml.safe_load(Path('params.yaml').read_text(encoding='utf-8'))
    uri = params['tracking']['uri'].rstrip('/')
    with urllib.request.urlopen(uri + '/health', timeout=5) as response:
        if response.status != 200:
            raise ValueError('MLflow Docker server is not ready')
    backup = Path('logs/offline-sweep.json')
    if not backup.exists():
        write(backup, read('metrics/sweep.json'))
    summary = read(backup)
    source_uri = 'sqlite:///' + (Path('logs/offline_mlflow.db').resolve().as_posix())
    source, target = MlflowClient(tracking_uri=source_uri), MlflowClient(tracking_uri=uri)
    parent = source.get_run(summary['parent_id'])
    ids = [summary['parent_id'], *[r['run_id'] for r in summary['rows']]]
    if len(ids) != 7 or any(source.get_run(rid).info.status != 'FINISHED' for rid in ids):
        raise ValueError('Expected a finished parent and six finished candidates')
    experiment = target.get_experiment_by_name(params['tracking']['experiment'])
    exp_id = experiment.experiment_id if experiment else target.create_experiment(params['tracking']['experiment'])
    mapping, evidence = {}, {}
    for old_id in ids:
        original = source.get_run(old_id)
        existing = target.search_runs([exp_id], filter_string=f"tags.offline_run_id = '{old_id}'")
        if len(existing) > 1:
            raise ValueError('Ambiguous import destination')
        tags = dict(original.data.tags, offline_run_id=old_id,
                    offline_tracking_uri=source_uri, import_method='MLflowClient: byte-verified artifacts, original metric timestamps')
        if 'mlflow.parentRunId' in tags:
            tags['mlflow.parentRunId'] = mapping[tags['mlflow.parentRunId']]
        if existing:
            destination = existing[0]
        else:
            destination = target.create_run(exp_id, start_time=original.info.start_time, tags=tags)
        new_id = destination.info.run_id
        mapping[old_id] = new_id
        if 'mlflow.log-model.history' in tags:
            history = json.loads(tags['mlflow.log-model.history'])
            for model in history:
                if model.get('run_id') == old_id:
                    model['run_id'] = new_id
            tags['mlflow.log-model.history'] = json.dumps(history)
        if destination.info.status != 'FINISHED':
            metrics = [m for key in original.data.metrics for m in source.get_metric_history(old_id, key)]
            target.log_batch(new_id, params=[Param(k, v) for k, v in original.data.params.items()],
                             metrics=[Metric(m.key, m.value, m.timestamp, m.step) for m in metrics],
                             tags=[RunTag(k, v) for k, v in tags.items()])
            with tempfile.TemporaryDirectory(prefix='hw67-import-') as temp:
                root = source.download_artifacts(old_id, '', temp)
                target.log_artifacts(new_id, root)
        with tempfile.TemporaryDirectory(prefix='hw67-verify-') as temp:
            left = Path(temp) / 'original'; left.mkdir()
            right = Path(temp) / 'server'; right.mkdir()
            expected = hashes(Path(source.download_artifacts(old_id, '', str(left))))
            actual = hashes(Path(target.download_artifacts(new_id, '', str(right))))
            # comparison.json — производный список IDs; при повторном импорте он уже перелинкован.
            immutable = set(expected) - ({'comparison.json'} if old_id == parent.info.run_id else set())
            if any(actual.get(path) != expected[path] for path in immutable):
                raise ValueError('Artifact SHA256 mismatch: ' + old_id)
            evidence[old_id] = dict(server_run_id=new_id, sha256={p: expected[p] for p in sorted(immutable)})
        target.set_terminated(new_id, status='FINISHED', end_time=original.info.end_time)
        print(f'Verified {old_id} -> {new_id}', flush=True)
    result = json.loads(json.dumps(summary))
    result['parent_id'] = mapping[summary['parent_id']]
    for row in result['rows']:
        row['run_id'] = mapping[row['run_id']]
    with tempfile.TemporaryDirectory(prefix='hw67-parent-') as temp:
        path = Path(temp) / 'comparison.json'
        write(path, dict(candidates=result['rows']))
        target.log_artifact(result['parent_id'], str(path))
    write('metrics/offline-import.json', dict(source=source_uri, target=uri, mapping=mapping,
                                             verified_artifacts=evidence, note='Original model packages are byte-identical; internal MLmodel.run_id retains source provenance'))
    write('metrics/sweep.json', result)
    print('All six real candidates copied; model bytes and measurements preserved.')


if __name__ == '__main__':
    main()
