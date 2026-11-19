"""Download the author's Pelican MAT without overwriting an existing raw file."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'http://wavelab.uwaterloo.ca/wp-content/uploads/2017/09/AscTec_Pelican_Flight_Dataset.mat'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'data/pelican/raw')
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    dest = args.directory / 'AscTec_Pelican_Flight_Dataset.mat'
    manifest = args.directory / 'provenance.json'
    if dest.exists():
        if not manifest.exists():
            raise RuntimeError('Existing raw file has no provenance; refusing to overwrite.')
        expected = json.loads(manifest.read_text(encoding='utf-8'))
        assert digest(dest) == expected['sha256'], 'Existing raw file checksum mismatch'
        print('Existing raw file verified:', dest, flush=True)
        return
    part = dest.with_suffix('.mat.part')
    started = time.monotonic()
    request = urllib.request.Request(URL, headers={'User-Agent': 'Pelican-research-experiment/1.0'})
    with urllib.request.urlopen(request, timeout=45) as response, part.open('xb') as f:
        expected_size = int(response.headers.get('Content-Length', '0'))
        final_url = response.url
        headers = dict(response.headers)
        count, last_report = 0, 0
        for chunk in iter(lambda: response.read(1024 * 1024), b''):
            f.write(chunk)
            count += len(chunk)
            if count - last_report >= 16 * 1024 * 1024:
                print(f'Downloaded {count / 1e6:.1f} / {expected_size / 1e6:.1f} MB', flush=True)
                last_report = count
            if time.monotonic() - started > 900:
                raise TimeoutError('Download exceeded 15 minutes; partial file retained.')
    if expected_size and part.stat().st_size != expected_size:
        raise RuntimeError('Download length mismatch; partial file retained.')
    with part.open('rb') as f:
        assert f.read(128).startswith(b'MATLAB 5.0 MAT-file'), 'Unexpected file format'
    sha = digest(part)
    part.rename(dest)
    metadata = dict(source_url=URL, final_url=final_url,
                    downloaded_utc=datetime.now(timezone.utc).isoformat(),
                    bytes=dest.stat().st_size, sha256=sha, response_headers=headers,
                    hash_status='Locally calculated; no publisher checksum was provided.',
                    source_repository='https://github.com/wavelab/pelican_dataset',
                    transport='Author-provided HTTP URL; not authenticated by TLS.')
    manifest.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({k: metadata[k] for k in ['bytes', 'sha256', 'downloaded_utc']}), flush=True)


if __name__ == '__main__':
    main()
