import re 

def convertNames(names):
    names = [re.sub("aenm", "assay_component_endpoint_name", name) for name in names]
    names = [re.sub("acnm", "assay_component_name", name) for name in names]
    names = [re.sub("anm", "assay_name", name) for name in names]
    names = [re.sub("asnm", "assay_source_name", name) for name in names]
    
    return names
