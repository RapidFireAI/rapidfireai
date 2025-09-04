"""This module contains the serialize functions for complexobjects."""

import base64

import dill


def encode_payload(payload: object) -> str:
    """Encode the payload for the database"""
    return base64.b64encode(dill.dumps(payload)).decode("utf-8")


def decode_db_payload(payload: str) -> object:
    """Decode the payload from the database"""
    return dill.loads(base64.b64decode(payload))
