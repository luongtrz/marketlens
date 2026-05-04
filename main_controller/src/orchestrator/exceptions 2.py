"""Pipeline-level exceptions."""


class PipelineError(Exception):
    """Raised when a required pipeline step fails and the run cannot continue."""
