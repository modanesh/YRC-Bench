import inspect


class BaseConfig:
    def __init__(self) -> None:
        """ Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods)."""
        self.init_member_classes(self)

    @staticmethod
    def init_member_classes(obj):
        # iterate over all attributes names
        for key in dir(obj):
            # disregard builtin attributes
            # if key.startswith("__"):
            if key == "__class__":
                continue
            # get the corresponding attribute object
            var = getattr(obj, key)
            # check if it the attribute is a class
            if inspect.isclass(var):
                # instantate the class
                i_var = var()
                # set the attribute to the instance instead of the type
                setattr(obj, key, i_var)
                # recursively init members of the attribute
                BaseConfig.init_member_classes(i_var)
    
    def as_string(self, specific_class=None):
        result = []
        self._str_helper(self, result, specific_class)
        return "\n".join(result)

    def _str_helper(self, obj, result, specific_class, prefix=""):
        for key in dir(obj):
            if key.startswith("__"):
                continue
            var = getattr(obj, key)
            if inspect.isclass(var):
                if specific_class and key != specific_class:
                    continue
                result.append(f"{prefix}{key}:")
                self._str_helper(var, result, specific_class, prefix="  ")
            else:
                result.append(f"{prefix}{key} = {var}")
