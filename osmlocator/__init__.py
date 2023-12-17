# __init__.py
__all__ = ['OsmLocator', 'evaluate', 'evaluateXY', 'getDefaultSetting']

from osmlocator.osm_locator import OsmLocator
from osmlocator.evaluator import evaluate, evaluateXY
from osmlocator.settings import getDefaultSetting
