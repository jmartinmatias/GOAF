# fix_pytorch.py
import sys
import types

def fix_torch_classes():
    """
    Fix PyTorch compatibility with Streamlit hot-reload.
    
    This prevents errors when Streamlit's file watcher tries to access
    torch._classes.__path__._path
    """
    # Only patch if torch is being used
    try:
        import torch
        
        # Create a dummy module that safely handles path access
        class DummyPathProperty:
            @property
            def _path(self):
                return []
        
        class DummyClassesModule(types.ModuleType):
            def __init__(self):
                super().__init__("torch._classes")
                self.__path__ = DummyPathProperty()
        
        # Check if we need to apply the patch
        if not hasattr(sys.modules.get('torch._classes', object()), '__path__') or \
           not hasattr(getattr(sys.modules.get('torch._classes', object()), '__path__', object()), '_path'):
            # Apply our patch
            sys.modules['torch._classes'] = DummyClassesModule()
            print("Applied PyTorch compatibility fix for Streamlit hot-reload")
            
    except ImportError:
        # Torch not installed, nothing to patch
        pass