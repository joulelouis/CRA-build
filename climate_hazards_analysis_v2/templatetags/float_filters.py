from django import template
import re

register = template.Library()

@register.filter
def to_float(value):
    """
    Convert a value to float, handling various formats and edge cases.
    Returns 0.0 if conversion fails.
    """
    if value is None or value == '' or value == 'N/A':
        return 0.0
    
    # If already a number
    if isinstance(value, (int, float)):
        return float(value)
    
    try:
        # Handle string values
        if isinstance(value, str):
            # Remove any non-numeric characters except decimal point and minus sign
            cleaned = re.sub(r'[^\d.-]', '', value)
            if cleaned and cleaned != '-':
                return float(cleaned)
    except (ValueError, TypeError):
        pass
    
    return 0.0