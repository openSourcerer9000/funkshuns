# -*- coding: utf-8 -*-

__author__ = """Sean Micek"""
__email__ = 'sean.micek@gmail.com'
__version__ = '0.1.0'

#drill down to the goods no matter where you're importing from
if __package__ is None or __package__ == '':
    # uses current directory visibility
    import funkshuns.funkshuns
else:
    # uses current package visibility
    from .funkshuns import funkshuns