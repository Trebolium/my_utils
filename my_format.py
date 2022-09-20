

# takes in an object with attributes via the '.' character and returns it as dictionary object
def attrs_to_dict(attr_object):
    return {attr: getattr(attr_object, attr) for attr in dir(attr_object) if not attr.startswith('_')}
