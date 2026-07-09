"""
Reusable UI components for the MLOps Dashboard.

Provides styled KPI cards, status badges, section headers, and empty states.
"""

import streamlit as st
from typing import Optional, Callable


def render_kpi_card(title: str, value: str, delta: Optional[str] = None, delta_type: str = "normal"):
    """
    Render a styled KPI card.
    
    Args:
        title: Card title
        value: Main value to display
        delta: Optional delta value (e.g., "+5.2%")
        delta_type: "normal", "inverse", or "off"
    """
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        {f'<div class="kpi-delta {delta_type}">{delta}</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)


def render_status_badge(status: str):
    """
    Render a status badge.
    
    Args:
        status: "Healthy", "Degraded", or "Missing Artifacts"
    """
    status_map = {
        "Healthy": ("success", "🟢"),
        "Degraded": ("warning", "🟡"),
        "Missing Artifacts": ("error", "🔴")
    }
    
    badge_class, icon = status_map.get(status, ("unknown", "⚪"))
    
    st.markdown(f"""
    <div class="status-badge {badge_class}">
        <span class="status-icon">{icon}</span>
        <span class="status-text">{status}</span>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str, subtitle: Optional[str] = None):
    """
    Render a styled section header.
    
    Args:
        title: Section title
        subtitle: Optional subtitle
    """
    subtitle_html = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ''
    
    st.markdown(f"""
    <div class="section-header">
        <h2 class="section-title">{title}</h2>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


def render_empty_state(
    message: str,
    action_button_text: Optional[str] = None,
    action_callback: Optional[Callable] = None,
    icon: str = "📭"
):
    """
    Render an empty state component.
    
    Args:
        message: Message to display
        action_button_text: Optional button text
        action_callback: Optional callback function for button
        icon: Icon emoji
    """
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-icon">{icon}</div>
        <div class="empty-state-message">{message}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if action_button_text and action_callback:
        if st.button(action_button_text, type="primary", use_container_width=True):
            action_callback()
            st.rerun()


def render_metric_with_color(value: float, threshold_good: float = 0.8, threshold_warn: float = 0.6):
    """
    Render a metric value with color coding.
    
    Args:
        value: Metric value (0-1)
        threshold_good: Threshold for "good" (green)
        threshold_warn: Threshold for "warning" (yellow)
    
    Returns:
        HTML string with colored metric
    """
    if value >= threshold_good:
        color_class = "metric-good"
    elif value >= threshold_warn:
        color_class = "metric-warn"
    else:
        color_class = "metric-bad"
    
    return f'<span class="metric-value {color_class}">{value:.4f}</span>'

