import logging
from functools import cached_property
from typing import List, Optional

from pinecone_text.hybrid import hybrid_convex_scale
from pinecone_text.sparse import BM25Encoder

from . import DenseRecordEncoder, OpenAIRecordEncoder
from .base import RecordEncoder
from canopy.knowledge_base.models import KBQuery, KBEncodedDocChunk, KBDocChunk
from canopy.models.data_models import Query

logger = logging.getLogger(__name__)


class HybridRecordEncoder(RecordEncoder):
    """
    HybridRecordEncoder is a subclass of RecordEncoder that generates sparse and dense vector representation of
    documents` chunks and textual queries.

    The dense representation generated by the `HybridRecordEncoder` is a list of floats in a given dimension.
    The sparse representation generated by the `HybridRecordEncoder` is a `SparseVector`.

    HybridRecordEncoder uses DenseRecordEncoder for dense encoding and BM25Encoder for sparse encoding.

    Alpha is a parameter that controls the weight of the dense vector in the hybrid representation.
    If alpha is 1, the query vector will be the dense vector. The default value of alpha is 0.5.

    For more information about the encoders see: https://github.com/pinecone-io/pinecone-text

    """  # noqa: E501

    _DEFAULT_COMPONENTS = {
        "dense_record_encoder": OpenAIRecordEncoder
    }

    def __init__(self,
                 dense_record_encoder: Optional[DenseRecordEncoder] = None,
                 alpha: float = 0.5,
                 bm_25_encoder_df_path: Optional[str] = None,
                 **kwargs):
        """
        Initialize the encoder.

        Args:
            dense_record_encoder: A DenseRecordEncoder to encode the text.
            alpha: The weight of the dense vector in the hybrid representation (between 0 and 1).
            bm_25_encoder_df_path: The path to the file that contains the document frequencies of the BM25Encoder.\
            You can create this file by fitting the BM25Encoder on a corpus of documents and calling `dump`\
            on the encoder.
            **kwargs: Additional arguments to pass to the RecordEncoder.
        """  # noqa: E501

        if alpha == 0:
            raise ValueError("Sparse only representation is not supported. "
                             "Alpha must be greater than 0.")

        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be between 0 (excluded) and 1 (included)")

        super().__init__(**kwargs)

        if dense_record_encoder:
            if not isinstance(dense_record_encoder, DenseRecordEncoder):
                raise TypeError(
                    f"dense_encoder must be an instance of DenseRecordEncoder, "
                    f"not {type(dense_record_encoder)}"
                )
            self._dense_record_encoder = dense_record_encoder
        else:
            default_dense_encoder = self._DEFAULT_COMPONENTS["dense_record_encoder"]
            self._dense_record_encoder = default_dense_encoder()

        self._bm_25_encoder_df_path = bm_25_encoder_df_path
        self._alpha = alpha

    @cached_property
    def _sparse_encoder(self) -> BM25Encoder:
        logger.info("Loading the document frequencies for the BM25Encoder...")
        if self._bm_25_encoder_df_path is None:
            encoder = BM25Encoder.default()
        else:
            encoder = BM25Encoder().load(self._bm_25_encoder_df_path)
        logger.info("Finished loading the document frequencies for the BM25Encoder.")
        return encoder

    def _encode_documents_batch(self,
                                documents: List[KBDocChunk]
                                ) -> List[KBEncodedDocChunk]:
        """
        Encode a batch of documents, takes a list of KBDocChunk and returns a list of KBEncodedDocChunk.

        Args:
            documents: A list of KBDocChunk to encode.
        Returns:
            encoded chunks: A list of KBEncodedDocChunk,
            with the `values` containing the generated dense vector and
            `sparse_values` containing the generated sparse vector.
        """  # noqa: E501

        chunks = self._dense_record_encoder.encode_documents(documents)
        sparse_values = self._sparse_encoder.encode_documents(
            [d.text for d in documents]
        )
        for chunk, sv in zip(chunks, sparse_values):
            chunk.sparse_values = sv
        return chunks

    def _encode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
        """
        Encode a batch of queries, takes a list of Query and returns a list of KBQuery.
        Args:
            queries: A list of Query to encode.
        Returns:
            encoded queries: A list of KBQuery, with the `values` containing the generated dense vector with the weight
            alpha and `sparse_values` containing the generated sparse vector with the weight (1 - alpha).
        """  # noqa: E501

        dense_queries = self._dense_record_encoder.encode_queries(queries)
        sparse_values = self._sparse_encoder.encode_queries([q.text for q in queries])

        scaled_values = [
            hybrid_convex_scale(dq.values, sv, self._alpha) for dq, sv in
            zip(dense_queries, sparse_values)
        ]

        return [q.copy(update=dict(values=v, sparse_values=sv)) for q, (v, sv) in
                zip(dense_queries, scaled_values)]

    @property
    def dimension(self) -> int:
        return self._dense_record_encoder.dimension

    async def _aencode_documents_batch(self,
                                       documents: List[KBDocChunk]
                                       ) -> List[KBEncodedDocChunk]:
        raise NotImplementedError

    async def _aencode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
        raise NotImplementedError