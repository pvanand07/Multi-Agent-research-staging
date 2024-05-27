# -*- coding: utf-8 -*-
"""Toggling component for Streamlit"""

from typing import Any, Callable, Optional
import sys
import os
import subprocess
import logging

logger = logging.getLogger(__name__)

import streamlit as st
import toml


def toggle(
    container: Optional[Any] = None,
    widget: str = 'slider',
    label: str = '',
    value: bool = False,
    key: Optional[str] = None,
    help: Optional[str] = None,
    on_change: Optional[Callable] = None,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
    *,
    disabled: bool = False,
) -> bool:
    """Add a toggle switch to Streamlit app.

    Args:
        container (any, optional): The Streamlit container. Defaults to `None`.
        widget (str, optional): The input widget. Defaults to `'slider'`.
        label (str, optional): The short label explaining what the toggle
    switch is for. Defaults to `''`.
        value (bool, optional): The preselected value on first renders.
    Defaults to `False`.
        key (str, optional): The unique key for the widget. Defaults to `None`,
    a key will be automatically generated.
        help (str, optional): The tooltip that gets displayed next to the
    toggle switch. Defaults to `None`.
        on_change (callable, optional): The callback invoked when the value of
    the toggle switch changes. Defaults to `None`.
        args (tuple, optional): The tuple of args to pass to the callback.
    Defaults to `None`.
        kwargs (dict, optional): The dictionary of kwargs to pass to the
    callback. Defaults to `None`.
        disabled (bool, optional, keyword-only): Whether the toggle switch is
    to be disabled. Defaults to `False`.

    Returns:
        (bool): The value of the toggle switch.
    """
    if container is None:
        container = st
    if widget == 'checkbox':
        return container.checkbox(
            label,
            value=value,
            key=key,
            help=help,
            on_change=on_change,
            args=args,
            kwargs=kwargs,
            disabled=disabled,
        )
    if widget == 'radio':
        return container.radio(
            label,
            options=('False', 'True'),
            index=int(value),
            key=key,
            help=help,
            on_change=on_change,
            args=args,
            kwargs=kwargs,
            disabled=disabled,
            horizontal=True,
            label_visibility='visible',
        ) == 'True'
    if widget == 'selectbox':
        return container.selectbox(
            label,
            options=('False', 'True'),
            value=str(value),
            key=key,
            help=help,
            on_change=on_change,
            args=args,
            kwargs=kwargs,
            disabled=disabled,
            label_visibility='visible',
        ) == 'True'
    if widget == 'slider':
        return container.select_slider(
            label,
            options=('False', 'True'),
            value=str(value),
            key=key,
            help=help,
            on_change=on_change,
            args=args,
            kwargs=kwargs,
            disabled=disabled,
            label_visibility='visible',
        ) == 'True'
    raise ValueError(f"widget '{widget}' is not supported")


def _config(path: str) -> str:
    """Return the Streamlit configuration file path given the path of Streamlit
    app. If the file does not exists, create it. If a `PermissionError`
    is raised, return the name of the Streamlit environment variable
    correspondingly.
    """
    path = os.path.join(os.path.dirname(path), '.streamlit/config.toml')
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, mode='w') as _:
                pass
    except PermissionError:
        path = 'STREAMLIT_THEME_BASE'
    logger.info(f'using the Streamlit configuration: {path}')
    return path


def _theme(
    config: str,
    value: bool,
    key: str = 'theming',
) -> None:
    """Overwrite the theme setting."""
    if key not in st.session_state:
        st.session_state[key] = value
    st.session_state[key] = not st.session_state[key]
    if os.path.exists(config):
        options = toml.load(config)
        options['theme'] = {
            'base': 'dark' if st.session_state[key] else 'light'
        }
        toml.dump(options, open(config, mode='w'))
        logger.warning(
            'overwrite the Streamlit configuration file, it takes effect only by rerunning'
        )
    else:
        os.environ[config] = 'dark' if st.session_state[key] else 'light'
        logger.warning(
            'overwrite the Streamlit environment variable, but it does not take effect'
        )


def theme(
    container: Optional[Any] = None,
    widget: Optional[str] = 'checkbox',
    label: str = 'Dark',
    value: bool = False,
    key: Optional[str] = None,
    help: Optional[str] = None,
    *,
    disabled: bool = False,
    path: Optional[str] = None,
) -> None:
    """Add a Streamlit Light and Dark theme toggle switch to Streamlit app.

    Args:
        container (any, optional): The Streamlit container. Defaults to `None`.
        widget (str, optional): The input widget. Defaults to `'slider'`.
        label (str, optional): The short label explaining what the toggle
    switch is for. Defaults to `'Dark'`.
        value (bool, optional): The preselected value on first renders.
    Defaults to `False`, the default Streamlit Light theme.
        key (str, optional): The unique key for the widget. Defaults to `None`,
    a key will be automatically generated.
        help (str, optional): The tooltip that gets displayed next to the
    toggle switch. Defaults to `None`.
        disabled (bool, optional, keyword-only): Whether the toggle switch is
    to be disabled. Defaults to `False`.
        path (str, optional, keyword-only): The path of Stramlit app. Defaults
    to `None`, the file path of the caller.

    Returns:
        (bool): The value of the toggle switch.
    """
    if path is None:
        path = sys._getframe(1).f_globals['__file__']
    return toggle(
        container=container,
        widget=widget,
        label=label,
        value=value,
        key=key,
        help=help,
        on_change=_theme,
        args=(_config(path), value),
        disabled=disabled,
    )
