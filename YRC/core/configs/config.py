
class ConfigDict:

    def __init__(self, **entries):
        self._entries = []
        rec_entries = {}
        for k, v in entries.items():
            if isinstance(v, dict):
                rv = ConfigDict(**v)
            else:
                rv = v
            rec_entries[k] = rv
            self._entries.append(k)
        self.__dict__.update(rec_entries)

    def __str_helper(self, depth):
        lines = []
        for k, v in self.__dict__.items():
            if k == "_entries":
                continue
            if isinstance(v, ConfigDict):
                v_str = v.__str_helper(depth + 1)
                lines.append("%s:\n%s" % (k, v_str))
            else:
                lines.append("%s: %r" % (k, v))
        indented_lines = ["    " * depth + l for l in lines]
        return "\n".join(indented_lines)

    def __str__(self):
        return "ConfigDict {\n%s\n}" % self.__str_helper(1)

    def __repr__(self):
        return "ConfigDict(%r)" % self.__dict__

    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            return None

    def __getitem__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            return None

    def __contains__(self, name):
        return name in self._entries

    def as_dict(self):
        ret = {}
        for k in self._entries:
            v = getattr(self, k)
            if isinstance(v, ConfigDict):
                rv = v.as_dict()
            else:
                rv = v
            ret[k] = rv
        return ret

    def clone(self):
        return self(**self.to_dict())
