"""Backward-compatible re-export; implementation lives in scad_model.py."""
from .scad_model import MultiDomainFENDModel, Trainer, orthogonality_penalty

__all__ = ['MultiDomainFENDModel', 'Trainer', 'orthogonality_penalty']
