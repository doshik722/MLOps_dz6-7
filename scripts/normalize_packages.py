"""MLflow 2.22 на Windows: переносимые пути внутри MLmodel, без изменения весов."""
import hashlib
import json
import tempfile
from pathlib import Path, PureWindowsPath

import mlflow
from mlflow import MlflowClient
import yaml


def main():
    config = yaml.safe_load(Path('params.yaml').read_text(encoding='utf-8'))
    mlflow.set_tracking_uri(__import__('os').environ.get('MLFLOW_TRACKING_URI', config['tracking']['uri']))
    client = MlflowClient()
    summary = json.loads(Path('metrics/sweep.json').read_text(encoding='utf-8'))
    proof = {}
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for row in summary['rows']:
        rid = row['run_id']
        with tempfile.TemporaryDirectory(prefix='hw67-package-') as temp:
            root = Path(client.download_artifacts(rid, 'model', temp))
            path = root / 'MLmodel'
            before = path.read_bytes()
            meta = yaml.safe_load(before.decode('utf-8'))
            flavor = meta['flavors']['python_function']
            flavor['model_code_path'] = PureWindowsPath(flavor['model_code_path']).name
            if not (root / flavor['model_code_path']).is_file():
                raise ValueError('Packaged Python model source is missing')
            for artifact in flavor.get('artifacts', {}).values():
                artifact['path'] = PureWindowsPath(artifact['path']).as_posix()
                resolved = (root / artifact['path']).resolve()
                if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
                    raise ValueError('Invalid packaged artifact path')
            path.write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding='utf-8')
            client.log_artifact(rid, str(path), 'model')
            client.set_tag(rid, 'package_paths_normalized', 'true')
            client.set_tag(rid, 'package_normalizer_sha256', digest)
            proof[rid] = dict(before_mlmodel_sha256=hashlib.sha256(before).hexdigest(),
                              after_mlmodel_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              note='Only MLmodel paths changed; adapter/model source/tokenizer bytes unchanged')
        print('Normalized package paths:', row['name'], flush=True)
    Path('metrics/package-paths.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
