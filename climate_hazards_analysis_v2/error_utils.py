import logging

logger = logging.getLogger(__name__)

def handle_sensitivity_param_error(context, error):
    """Handle errors while processing sensitivity parameters."""
    logger.error(f"Error processing sensitivity parameters: {error}")
    context['error'] = f"Error processing parameters: {error}"