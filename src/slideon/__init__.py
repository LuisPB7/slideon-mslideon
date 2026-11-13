# Public package surface: the adapter class and two checkpoint-loading helpers.
# The Python package is named slideon even though the distribution installed by
# pip retains the upstream name slideon-mslideon.
#
# SlideOn below is a Sentence Transformers InputModule. It is distinct from
# modeling_slideon.SlideOnModel, the Transformers neural network inside it.
from .models import SlideOn
from .loading import from_sae, from_pretrained

# Export these names for users importing from slideon. Loading helpers return
# a complete SparseEncoder wrapper, rather than a bare SAE or backbone.
__all__ = ["SlideOn", "from_sae", "from_pretrained"]
