# Expose the generative models at the package level for clean imports
from .normal import normal_model
from .student_t import student_t_model
from .contaminated import contaminated_normal_model
from .bootstrap import bayesian_bootstrap_mean, bayesian_bootstrap_trimmed

__all__ = [
    "normal_model",
    "student_t_model",
    "contaminated_normal_model",
    "bayesian_bootstrap_mean",
    "bayesian_bootstrap_trimmed",
]