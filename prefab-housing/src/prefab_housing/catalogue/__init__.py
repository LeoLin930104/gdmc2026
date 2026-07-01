"""Catalogue subsystem: pod types, face categories, template synthesis.

The catalogue is *the* central data table. Tile compatibility, scoring, and
materialisation all read from it. Hot loops index by integer ``pod_type_id``
and ``template_id``; the named enums are only for boundary readability.
"""
