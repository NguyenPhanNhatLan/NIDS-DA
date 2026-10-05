"""Read Kaggle CSV batches over HTTP without storing dataset files locally."""
from __future__ import annotations

import argparse
import io
from itertools import chain
from urllib.parse import quote

import pandas as pd
import requests
from stream_unzip import stream_unzip

DEFAULT_DATASET = 'yasserhessein/cic-unsw-nb15-augmented-dataset'


class ChunkReader(io.RawIOBase):
    """Bounded-memory file interface for an iterator of byte chunks."""
    def __init__(self, chunks):
        super().__init__()
        self.chunks = iter(chunks)
        self.pending = memoryview(b'')

    def readable(self):
        return True

    def readinto(self, buffer):
        while not self.pending:
            try:
                self.pending = memoryview(next(self.chunks))
            except StopIteration:
                return 0
        size = min(len(buffer), len(self.pending))
        buffer[:size] = self.pending[:size]
        self.pending = self.pending[size:]
        return size


def iter_kaggle_csv(filename='CICFlowMeter.csv', dataset=DEFAULT_DATASET,
                    chunksize=50000, **read_csv_options):
    """Yield DataFrames; consume inside try/finally and close on early exit.

    Transfers bytes over the network, but writes neither CSV nor ZIP to disk.
    Each call starts a fresh remote read. Does not infer or merge labels.
    """
    if chunksize < 1:
        raise ValueError('chunksize must be positive')
    parts = dataset.split('/')
    if len(parts) != 2 or not all(parts):
        raise ValueError('dataset must be owner/dataset-slug')
    url = ('https://www.kaggle.com/api/v1/datasets/download/'
           + '/'.join(quote(part, safe='') for part in parts)
           + '/' + quote(filename, safe=''))
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if 'text/html' in response.headers.get('Content-Type', '').lower():
            raise RuntimeError('Kaggle returned HTML instead of data; check dataset access.')
        chunks = (chunk for chunk in response.iter_content(65536) if chunk)
        first = next(chunks, b'')
        if not first:
            raise RuntimeError('Kaggle returned an empty file.')
        chunks = chain((first,), chunks)
        if first.startswith(b'PK'):
            matched = False
            for name, _, contents in stream_unzip(chunks):
                member = name.decode('utf-8')
                if member == filename or member.rsplit('/', 1)[-1] == filename:
                    matched = True
                    with io.BufferedReader(ChunkReader(contents)) as source:
                        with pd.read_csv(source, chunksize=chunksize, **read_csv_options) as reader:
                            yield from reader
                else:
                    for _ in contents:
                        pass
            if not matched:
                raise FileNotFoundError(f'{filename} not found in Kaggle ZIP.')
        else:
            with io.BufferedReader(ChunkReader(chunks)) as source:
                with pd.read_csv(source, chunksize=chunksize, **read_csv_options) as reader:
                    yield from reader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--file', default='CICFlowMeter.csv')
    parser.add_argument('--chunksize', type=int, default=50000)
    parser.add_argument('--preview-rows', type=int, default=5)
    args = parser.parse_args()
    if args.preview_rows < 1:
        parser.error('--preview-rows must be positive')
    batches = iter_kaggle_csv(args.file, args.dataset, args.chunksize)
    try:
        batch = next(batches)
        print('Columns:', batch.columns.tolist())
        print(batch.head(args.preview_rows).to_string(index=False))
        print('Preview only; no dataset files saved locally.')
    finally:
        batches.close()


if __name__ == '__main__':
    main()
