"""Baixa apenas title.basics e title.ratings, mantendo os arquivos compactados."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import urllib.request

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    raw = args.project_root / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    out = args.project_root / 'results/imdb'
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'download_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    for name in ['title.basics.tsv.gz', 'title.ratings.tsv.gz']:
        destination = raw / name
        url = 'https://datasets.imdbws.com/' + name
        if destination.exists():
            print(f'Arquivo existente preservado: {name}', flush=True)
            if name not in manifest:
                manifest[name] = {'url': url, 'downloaded_at_utc': None,
                                  'provenance': 'existing_local_file'}
        else:
            temporary = destination.with_suffix(destination.suffix + '.part')
            request = urllib.request.Request(url, headers={'User-Agent': 'AcademicMovieMining/1.0'})
            print(f'Baixando {name}', flush=True)
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open('wb') as stream:
                size, next_report = 0, 64 * 1024 * 1024
                for block in iter(lambda: response.read(1024*1024), b''):
                    stream.write(block)
                    size += len(block)
                    if size >= next_report:
                        print(f'{name}: {size/1024**2:.0f} MiB recebidos', flush=True)
                        next_report += 64*1024*1024
                server_headers = {key: response.headers.get(key) for key in
                                  ['Last-Modified','ETag','Content-Length']}
            if server_headers['Content-Length'] is not None and size != int(server_headers['Content-Length']):
                raise ValueError('Download incompleto: tamanho diferente do informado pelo servidor.')
            temporary.replace(destination)
            manifest[name] = {'url': url, 'downloaded_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                              'server_headers': server_headers}
        digest = hashlib.sha256()
        with destination.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        manifest[name].update(bytes=destination.stat().st_size, sha256=digest.hexdigest())
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'{name}: concluído ({destination.stat().st_size/1024**2:.1f} MiB)', flush=True)

if __name__ == '__main__':
    main()
