# streamlit_torch_fix.py
import sys
import types

def apply_fix():
    """
    Apply a simple fix for the PyTorch-Streamlit compatibility issue.
    This just patches torch._classes to safely handle __path__._path access.
    """
    try:
        # Only attempt the fix if torch is installed
        import torch
        
        # Create a safer version of the __path__ object
        class SafePath:
            @property
            def _path(self):
                return []
        
        # Patch torch._classes.__path__
        if 'torch._classes' in sys.modules:
            # The module exists, just patch its __path__
            sys.modules['torch._classes'].__path__ = SafePath()
            print("✅ Applied PyTorch compatibility fix for Streamlit")
        else:
            # Need to create a new module
            class SafeClassesModule(types.ModuleType):
                def __init__(self):
                    super().__init__("torch._classes")
                    self.__path__ = SafePath()
            
            # Replace the module in sys.modules
            sys.modules['torch._classes'] = SafeClassesModule()
            print("✅ Created safe PyTorch module for Streamlit compatibility")
        
    except ImportError:
        # Torch not installed, nothing to patch
        pass
    except Exception as e:
        print(f"⚠️ Warning: Could not apply PyTorch fix: {e}")