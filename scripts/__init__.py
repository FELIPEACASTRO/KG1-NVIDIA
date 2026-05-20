"""KG1 local scripts package.

This file intentionally makes the repository ``scripts`` directory a regular
package. Remote job images can contain unrelated third-party packages named
``scripts``; without this file, namespace-package resolution can make weak
evaluation imports resolve to the wrong package.
"""
