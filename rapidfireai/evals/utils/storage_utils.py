"""Cloud storage abstraction for multimodal RAG artifacts.

This module provides :class:`CloudStorage`, a thin wrapper over either
Amazon S3 or Google Cloud Storage. It is used by the multimodal
unstructured-loader notebook to offload large per-document artifacts
(images, table HTML) out of the vector DB record and into object
storage, keeping vector metadata well under provider size limits.

Object key layout::

    {prefix}/{document_type}/{rf_doc_id}/document

The filename is literally ``document`` (no extension); the cloud
object's ``Content-Type`` is set based on ``document_type``.

Authentication relies on each SDK's default credential chain:
    - S3: ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` env vars,
      shared credentials file, IAM role, etc.
    - GCS: ``GOOGLE_APPLICATION_CREDENTIALS`` env var, ADC, attached
      service account, etc.

Backend SDKs are imported lazily so callers only need whichever one
they use (``pip install boto3`` or ``pip install google-cloud-storage``).
"""

from __future__ import annotations

import base64
from typing import Literal
from urllib.parse import urlparse

Backend = Literal["s3", "gcs"]

_IMAGE_CONTENT_TYPE = "image/jpeg"
_HTML_CONTENT_TYPE = "text/html; charset=utf-8"
_TEXT_CONTENT_TYPE = "text/plain; charset=utf-8"
_DOCUMENT_FILENAME = "document"


class CloudStorage:
    """Read/write artifacts to S3 or GCS using a uniform interface.

    Parameters
    ----------
    backend:
        Either ``"s3"`` or ``"gcs"``.
    bucket:
        Name of the bucket to read/write from. Must already exist.
    prefix:
        Top-level key prefix used for all objects. Defaults to
        ``"artifacts"`` so keys look like ``artifacts/image/<uuid>/document``.
    """

    def __init__(self, backend: Backend, bucket: str, prefix: str = "artifacts") -> None:
        if backend not in ("s3", "gcs"):
            raise ValueError(f"backend must be 's3' or 'gcs', got {backend!r}")
        if not bucket:
            raise ValueError("bucket must be a non-empty string")

        self.backend: Backend = backend
        self.bucket = bucket
        self.prefix = prefix.strip("/")

        self._s3_client = None
        self._gcs_bucket = None

        if backend == "s3":
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "The 's3' backend requires the 'boto3' package. "
                    "Install it with: pip install boto3"
                ) from None

            self._s3_client = boto3.client("s3")
        else:
            try:
                from google.cloud import storage
            except ImportError:
                raise ImportError(
                    "The 'gcs' backend requires the 'google-cloud-storage' package. "
                    "Install it with: pip install google-cloud-storage"
                ) from None

            self._gcs_bucket = storage.Client().bucket(bucket)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def put_image(self, doc_id: str, image_base64: str) -> str:
        """Upload a base64-encoded image and return its native URI.

        The base64 string is decoded to raw bytes before upload, so the
        stored object is a real image blob (Content-Type ``image/jpeg``).
        """
        if not isinstance(image_base64, str) or not image_base64:
            raise ValueError("image_base64 must be a non-empty base64 string")
        data = base64.b64decode(image_base64)
        key = self._build_key("image", doc_id)
        self._upload_bytes(key, data, content_type=_IMAGE_CONTENT_TYPE)
        return self._native_uri(key)

    def put_html(self, doc_id: str, html: str) -> str:
        """Upload table HTML as a UTF-8 ``.html`` blob and return its URI."""
        if not isinstance(html, str) or not html:
            raise ValueError("html must be a non-empty string")
        data = html.encode("utf-8")
        key = self._build_key("table", doc_id)
        self._upload_bytes(key, data, content_type=_HTML_CONTENT_TYPE)
        return self._native_uri(key)

    def put_text(self, doc_id: str, text: str) -> str:
        """Upload raw text as a UTF-8 plain-text blob and return its URI.

        Used by multimodal RAG to offload the original full text of a
        Unstructured CompositeElement out of the vector DB record. After
        summarization, the in-metadata ``text_source`` is replaced with the
        returned URI so the post-split children sharing this URI don't bloat
        per-upsert payload size.
        """
        if not isinstance(text, str) or not text:
            raise ValueError("text must be a non-empty string")
        data = text.encode("utf-8")
        key = self._build_key("text", doc_id)
        self._upload_bytes(key, data, content_type=_TEXT_CONTENT_TYPE)
        return self._native_uri(key)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get_image_base64(self, uri: str) -> str:
        """Download an image artifact and return it as a base64 string."""
        return base64.b64encode(self.read_bytes(uri)).decode("ascii")

    def get_html(self, uri: str) -> str:
        """Download a table HTML artifact and return it as a string."""
        return self.read_bytes(uri).decode("utf-8")

    def get_text(self, uri: str) -> str:
        """Download a text artifact and return it as a string."""
        return self.read_bytes(uri).decode("utf-8")

    def read_bytes(self, uri: str) -> bytes:
        """Download the raw bytes of any artifact identified by ``uri``."""
        bucket, key = self._parse_uri(uri)
        if self.backend == "s3":
            response = self._s3_client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        return self._gcs_client_for(bucket).blob(key).download_as_bytes()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def to_https_url(self, uri: str) -> str:
        """Convert an ``s3://`` or ``gs://`` URI to its HTTPS equivalent.

        Note: the returned URL is only fetchable by anonymous clients if
        the underlying object permits public reads. This helper performs
        no ACL changes; it is purely a URL transformation.
        """
        scheme, bucket, key = self._split_uri(uri)
        if scheme == "s3":
            return f"https://{bucket}.s3.amazonaws.com/{key}"
        return f"https://storage.googleapis.com/{bucket}/{key}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_key(self, document_type: str, doc_id: str) -> str:
        if not doc_id:
            raise ValueError("doc_id must be a non-empty string")
        return f"{self.prefix}/{document_type}/{doc_id}/{_DOCUMENT_FILENAME}"

    def _native_uri(self, key: str) -> str:
        scheme = "s3" if self.backend == "s3" else "gs"
        return f"{scheme}://{self.bucket}/{key}"

    def _upload_bytes(self, key: str, data: bytes, *, content_type: str) -> None:
        if self.backend == "s3":
            self._s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
        else:
            blob = self._gcs_bucket.blob(key)
            blob.upload_from_string(data, content_type=content_type)

    @staticmethod
    def _split_uri(uri: str) -> tuple[str, str, str]:
        parsed = urlparse(uri)
        if parsed.scheme not in ("s3", "gs"):
            raise ValueError(f"Unsupported URI scheme: {uri!r}")
        if not parsed.netloc:
            raise ValueError(f"URI is missing a bucket: {uri!r}")
        key = parsed.path.lstrip("/")
        if not key:
            raise ValueError(f"URI is missing an object key: {uri!r}")
        return parsed.scheme, parsed.netloc, key

    def _parse_uri(self, uri: str) -> tuple[str, str]:
        scheme, bucket, key = self._split_uri(uri)
        expected_scheme = "s3" if self.backend == "s3" else "gs"
        if scheme != expected_scheme:
            raise ValueError(
                f"URI scheme {scheme!r} does not match backend {self.backend!r}"
            )
        return bucket, key

    def _gcs_client_for(self, bucket: str):
        if bucket == self.bucket:
            return self._gcs_bucket
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "The 'gcs' backend requires the 'google-cloud-storage' package. "
                "Install it with: pip install google-cloud-storage"
            ) from None

        return storage.Client().bucket(bucket)
