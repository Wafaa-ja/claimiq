"""Formatting utility functions for ClaimIQ."""


def format_frequency(value: float) -> str:
    """Format a claim frequency value to 4 decimal places."""
    return f"{value:.4f}"


def format_currency(value: float, currency: str = "SAR") -> str:
    """Format a monetary value with thousands separators."""
    return f"{currency} {value:,.2f}"


def format_percentage(value: float) -> str:
    """Format a proportion as a percentage string."""
    return f"{value * 100:.1f}%"


def format_number(value: float, decimals: int = 4) -> str:
    """Format a numeric value to the specified number of decimal places."""
    return f"{value:.{decimals}f}"
