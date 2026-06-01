"""Long-tail hunter: generate search strategies that route around popularity priors."""
from .topic import Topic, TopicKind
from .query import Query, SearchPlan
from .hunter import plan

__all__ = ["Topic", "TopicKind", "Query", "SearchPlan", "plan"]
__version__ = "0.1.0"
