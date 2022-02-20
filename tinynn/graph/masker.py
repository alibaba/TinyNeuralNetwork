from copy import deepcopy
import torch


class Masker(object):
    """Manage mask of module"""

    def __init__(self, module, unique_name):
        self.module = module
        self.unique_name = unique_name
        self.masks = {}
        self.enabled = False

        # del old hook
        if hasattr(self.module, "masker"):
            self.module.masker.hook.remove()

        # add new hook
        if isinstance(self.module, torch.nn.Module):
            self.hook = self.module.register_forward_pre_hook(self)

        setattr(module, "masker", self)

    def __call__(self, module, inputs):
        """Apply mask when module forward"""
        if not self.enabled:
            return

        for tensor_name, mask in self.masks.items():
            t = getattr(module, tensor_name)
            t.data = t.data.mul_(mask)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def get_mask(self, tensor_name):
        return self.masks.get(tensor_name, None)

    def register_mask(self, tensor_name, mask):
        """Register a mask to the module, a module can have multiple masks"""
        if self.masks.get(tensor_name, None) is not None:
            del self.masks[tensor_name]

        self.masks[tensor_name] = torch.nn.Parameter(mask, requires_grad=False)
        setattr(self.module, f"{tensor_name}_mask", self.masks[tensor_name])

    def unregister_all(self):
        """Unregister all masks"""
        if isinstance(self.module, torch.nn.Module):
            self.hook.remove()

        for tensor_name, mask in self.masks.items():
            delattr(self.module, f"{tensor_name}_mask")
        self.masks.clear()

    def serialization(self):
        return self.unique_name, [self.masks]

    def deserialization(self, value):
        self.masks = value[0]


class ChannelMasker(Masker):
    """Channel-wise module masking"""

    def __init__(self, module, unique_name):
        super(ChannelMasker, self).__init__(module, unique_name)

        # Input channel to be deleted
        self.in_remove_idx = None

        # Output channel to be deleted
        self.ot_remove_idx = None

        # Custom channel to be deleted
        self.custom_remove_idx = None

    def set_in_remove_idx(self, in_remove_idx):
        if self.in_remove_idx is not None:
            self.in_remove_idx.extend(deepcopy(in_remove_idx))
            self.in_remove_idx.sort()
        else:
            self.in_remove_idx = deepcopy(in_remove_idx)

    def set_ot_remove_idx(self, ot_remove_idx):
        if self.ot_remove_idx is not None:
            self.ot_remove_idx.extend(deepcopy(ot_remove_idx))
            self.ot_remove_idx.sort()
        else:
            self.ot_remove_idx = deepcopy(ot_remove_idx)

    def set_custom_remove_idx(self, custom_remove_idx):
        if self.custom_remove_idx is not None:
            self.custom_remove_idx.extend(deepcopy(custom_remove_idx))
            self.custom_remove_idx.sort()
        else:
            self.custom_remove_idx = deepcopy(custom_remove_idx)

    def serialization(self):
        return self.unique_name, [self.masks, self.in_remove_idx, self.ot_remove_idx]

    def deserialization(self, value):
        self.masks = value[0]
        self.in_remove_idx = value[1]
        self.ot_remove_idx = value[2]
