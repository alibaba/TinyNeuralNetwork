import importlib


def try_import(pkg):
    try:
        return importlib.import_module(pkg)
    except Exception:
        return None


def register_on(name):
    def wrapper(func):
        name_parts = name.split('.')
        pkg = '.'.join(name_parts[:-1])
        prop = name_parts[-1]
        mod = try_import(pkg)

        if mod is not None:
            cls = getattr(mod, prop)
            setattr(cls, '__repr__', func)

    return wrapper


@register_on('transformers.models.opt.modeling_opt.OPTLearnedPositionalEmbedding')
def __OPTLearnedPositionalEmbedding__repr__(self):
    return f"{self.__class__.__name__}({self.num_embeddings - self.offset}, {self.embedding_dim})"
