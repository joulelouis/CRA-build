
from django import template

register = template.Library()

@register.filter(name='to_float')
def to_float(value):
    """
    Converts the value to a float. Returns 0.0 if conversion fails.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0