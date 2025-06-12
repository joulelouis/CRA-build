
from django import template
import re

register = template.Library()

@register.filter(name="to_float")
def to_float(value):

    if value is None:
        return 0.0

    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            try:
                return float(match.group(0))
            except (ValueError, TypeError):
                pass

    return 0.0