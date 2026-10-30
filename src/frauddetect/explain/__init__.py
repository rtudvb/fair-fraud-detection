"""Knowledge discovery (RQ5, experiment E8): SHAP summaries and surrogate-tree rules."""

from frauddetect.explain.rules import SurrogateRules, fit_surrogate, leaf_rules, one_hot_frame

__all__ = ["SurrogateRules", "fit_surrogate", "leaf_rules", "one_hot_frame"]
