"""Asynchronous DataPipeline using a secondary CUDA stream.

While the model is computing forward+backward on batch i, a background thread
prefetches batch i+1 on a separate CUDA stream.

Stream synchronization protocol:
  1. After graph.advance(i): prefetch_stream.wait_stream(main_stream)
     → ensures prefetch sees graph state AFTER advance(i)
  2. After prefetch completes: main_stream.wait_stream(prefetch_stream)
     → ensures forward(i+1) sees fully-prepared data tensors

Thread safety:
  - graph.recent() runs only in the prefetch thread
  - graph.advance() runs only in the main thread
  - get() joins the thread before advance(i+1) is called
  → no concurrent reads/writes to the ring buffer

Expected gain (DyGFormer B=200): sampling is ~4% of wall time, so max speedup
is ~1.04x end-to-end. Larger gains for models with heavy co-occurrence or
larger ring buffers relative to forward time.
"""

from __future__ import annotations

import threading
from typing import Optional

import torch

from tgengine.core.batch import RawBatch, PreparedBatch
from tgengine.core.gather_spec import GatherSpec
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.pipeline import DataPipeline
from tgengine.pipeline.negatives import NegativeStrategy


class AsyncDataPipeline:
    """Double-buffered pipeline that prefetches batch i+1 on a secondary CUDA stream.

    Usage pattern (in training loop):
        pipe = AsyncDataPipeline(spec, graph, neg_strat)

        pipe.start_prefetch(batches[0])          # prime: first batch
        for i, raw in enumerate(batches[:-1]):
            rb, prepared = pipe.get()            # wait for current prefetch

            opt.zero_grad()
            out = model(prepared)
            out.loss.backward()
            opt.step()
            graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)  # BEFORE next prefetch
            model.evolve(...)

            pipe.start_prefetch(batches[i + 1])  # start AFTER advance

        rb, prepared = pipe.get()                # last batch
        # process last batch...

    IMPORTANT: always call start_prefetch() AFTER graph.advance() for the
    current batch so the prefetch sees the updated graph state.
    """

    def __init__(
        self,
        spec: GatherSpec,
        graph: TemporalGraph,
        neg_strat: NegativeStrategy,
    ):
        self._pipeline = DataPipeline(spec, graph)
        self._neg_strat = neg_strat
        self._graph = graph
        self._prefetch_stream = torch.cuda.Stream()
        self._thread: Optional[threading.Thread] = None
        self._result: Optional[tuple[RawBatch, PreparedBatch]] = None
        self._exc: Optional[BaseException] = None

    @property
    def spec(self):
        return self._pipeline.spec

    @property
    def graph(self):
        return self._pipeline.graph

    def start_prefetch(self, raw_batch: RawBatch) -> None:
        """Launch prefetch in a background thread on the secondary CUDA stream.

        Must be called AFTER graph.advance() for the current batch.
        """
        # Ensure prefetch stream sees all pending work on the main stream
        # (including advance() that was just called).
        self._prefetch_stream.wait_stream(torch.cuda.current_stream())
        self._result = None
        self._exc = None

        prefetch_stream = self._prefetch_stream  # local ref for thread closure
        neg_strat = self._neg_strat
        pipeline = self._pipeline

        def _fetch():
            try:
                with torch.cuda.stream(prefetch_stream):
                    neg = neg_strat.sample(
                        raw_batch.src, raw_batch.dst, raw_batch.time, pipeline.graph,
                        raw_batch.edge_indices,
                    )
                    rb = RawBatch(
                        src=raw_batch.src, dst=raw_batch.dst, time=raw_batch.time,
                        edge_feat=raw_batch.edge_feat, neg=neg,
                        edge_indices=raw_batch.edge_indices,
                        neg_src=raw_batch.neg_src,
                    )
                    prepared = pipeline.prepare(rb)
                    self._result = (rb, prepared)
            except BaseException as exc:
                self._exc = exc

        self._thread = threading.Thread(target=_fetch, daemon=True)
        self._thread.start()

    def get(self) -> tuple[RawBatch, PreparedBatch]:
        """Block until prefetch is done; return (raw_batch_with_neg, prepared_batch).

        Synchronizes the prefetch stream into the main stream so all tensor
        data is safe to use on the default stream.
        """
        if self._thread is not None:
            self._thread.join()
            self._thread = None

        if self._exc is not None:
            raise self._exc

        # Main stream waits for any remaining prefetch-stream GPU work.
        torch.cuda.current_stream().wait_stream(self._prefetch_stream)

        result = self._result
        self._result = None
        return result  # type: ignore[return-value]

    def prepare(self, raw_batch: RawBatch) -> PreparedBatch:
        """Synchronous fallback — same interface as DataPipeline.prepare()."""
        return self._pipeline.prepare(raw_batch)
