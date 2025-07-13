"""
ComfyUI-ThinkSound: Video-to-Audio Generation using ThinkSound
Converts silent videos to audio using AI-generated sound based on visual content and text prompts.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = "./web"