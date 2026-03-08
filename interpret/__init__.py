"""
interpret: Model interpretability tools for genomic sequence models.
"""

__all__: list[str] = []

# Interpretability modules depend on gReLU — import only if available
try:
    from interpret.attribution import get_attributions_for_element, attribution_native_only
    __all__ += ["get_attributions_for_element", "attribution_native_only"]
except ImportError:
    pass

try:
    from interpret.ism import run_ism, predict_sequence
    __all__ += ["run_ism", "predict_sequence"]
except ImportError:
    pass

try:
    from interpret.modisco import run_modisco
    __all__ += ["run_modisco"]
except ImportError:
    pass
